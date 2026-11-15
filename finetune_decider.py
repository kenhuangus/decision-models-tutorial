import os, sys, time, json
from pathlib import Path
import torch
import torch.nn.functional as F

# Model snapshot directory (configurable via DECIDER_MODEL_PATH)
default_snap = Path.home() / ".cache" / "huggingface" / "hub" / "models--perplexity-ai--pplx-decider-flash-v1.2" / "snapshots" / "9ff7cf9838e59df461c276b7699d56b244d2af58"
snap = Path(os.environ.get("DECIDER_MODEL_PATH", str(default_snap)))
sys.path.insert(0, str(snap / "source" / "src"))
from pplx_decider.model import Decider
from pplx_decider.protocol import left_pad

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Decider on {device} for fine-tuning...")
decider = Decider(snap, device=device)

# 1. Domain Dataset: Enterprise Cyber Incident Triage & Automated Action
training_data = [
    # Class 0: containment
    {"state": "Active Cobalt Strike beaconing detected from finance workstation 10.0.4.12 to external IP 198.51.100.23. C2 traffic confirmed.", "target": [1.0, 0.0, 0.0]},
    {"state": "Ransomware encryption lockfile .locked detected spreading across SMB share \\\\nas01\\accounting.", "target": [1.0, 0.0, 0.0]},
    {"state": "Compromised admin credentials used to dump NTDS.dit active directory database from domain controller DC01.", "target": [1.0, 0.0, 0.0]},
    {"state": "Reverse shell spawned from internal web server /var/www/uploads/shell.php connecting to suspicious port 4444.", "target": [1.0, 0.0, 0.0]},
    
    # Class 1: investigation
    {"state": "Unusual spike in outbound DNS tunneling queries observed on developer machine 10.0.12.88. Signature is unconfirmed.", "target": [0.0, 1.0, 0.0]},
    {"state": "New cron job added under /etc/cron.d/sync_backup by unknown service account. Script contains base64 string.", "target": [0.0, 1.0, 0.0]},
    {"state": "User logged in from Singapore 10 minutes after logging in from San Francisco. Potential impossible travel.", "target": [0.0, 1.0, 0.0]},
    {"state": "Process injection attempt reported by EDR against lsass.exe, but operation was blocked by kernel filter.", "target": [0.0, 1.0, 0.0]},

    # Class 2: false_positive
    {"state": "Vulnerability scanner Nessus running scheduled port audit from authorized security subnet 10.99.0.0/24.", "target": [0.0, 0.0, 1.0]},
    {"state": "System administrator executing mass curl scripts to test internal API latency following cluster migration.", "target": [0.0, 0.0, 1.0]},
    {"state": "Backup agent Veeam reading large volume of VHDX files during approved midnight backup window.", "target": [0.0, 0.0, 1.0]},
    {"state": "Static analysis alert for hardcoded key in test_mock_credentials.py under unit test directory.", "target": [0.0, 0.0, 1.0]}
]

val_data = [
    {"state": "Lateral movement attempt using PsExec detected from dev machine 10.0.5.15 toward payroll database.", "expected": "containment", "target": [1.0, 0.0, 0.0]},
    {"state": "Anomalous PowerShell execution downloading remote raw pastebin text, but execution failed with 404 error.", "expected": "investigation", "target": [0.0, 1.0, 0.0]},
    {"state": "Security team conducting authorized red-team phishing exercise from verified test campaign domain.", "expected": "false_positive", "target": [0.0, 0.0, 1.0]}
]

question = {
    "triage": {
        "type": "choice",
        "instructions": "Determine the appropriate incident response action.",
        "criteria": {
            "containment": "Immediate automated isolation of host, session revocation, network block.",
            "investigation": "Pull memory artifacts, inspect network pcaps, triage alert.",
            "false_positive": "Benign administrative tool, scheduled backup, authorized test."
        }
    }
}

print("Running baseline evaluation before fine-tuning...")
baseline_scores = []
for item in val_data:
    resp = decider.decide({"state": item["state"], "questions": question})
    choice = resp["answers"]["triage"]["choice"]
    prob = resp["answers"]["triage"]["probabilities"][choice]
    baseline_scores.append({"state": item["state"][:60], "expected": item["expected"], "pred": choice, "prob": round(prob, 4)})
    print(f"  [Baseline] Expected: {item['expected']:<14} | Predicted: {choice:<14} | Prob: {prob:.4f}")

# 2. Fine-tuning Pipeline
print("\nInitiating Decision Head Fine-Tuning Pipeline...")
decider.readout.requires_grad_(True)
decider.readout.train()

optimizer = torch.optim.AdamW(decider.readout.parameters(), lr=1e-3, weight_decay=1e-4)
brier_weight = 0.15

epochs = 12
losses = []

for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    epoch_ce = 0.0
    epoch_brier = 0.0
    
    for item in training_data:
        optimizer.zero_grad()
        
        req = {"state": item["state"], "questions": question}
        chunk = decider.encoder.encode(req)
        chunks = [chunk]
        inputs = {key: value.to(decider.device) for key, value in left_pad(chunks, decider.encoder.pad_id).items()}
        
        with torch.no_grad():
            hidden = decider.backbone(**inputs, use_cache=False).last_hidden_state
        
        offsets = [inputs["input_ids"].shape[1] - chunk.length for chunk in chunks]
        rows = torch.cat([torch.full((len(c.keys),), i) for i, c in enumerate(chunks)]).to(decider.device)
        cols = torch.cat([c.markers + offsets[i] for i, c in enumerate(chunks)]).to(decider.device)
        
        logits = decider.readout(hidden[rows, cols])[0, :3]  # first question, 3 options
        
        target_tensor = torch.tensor(item["target"], device=decider.device, dtype=torch.float32)
        
        # Decision Loss = Cross-Entropy + Brier Calibration Penalty
        log_prob = F.log_softmax(logits / decider.temperature, dim=-1)
        ce_loss = -(target_tensor * log_prob).sum()
        brier_loss = (log_prob.exp() - target_tensor).square().sum()
        total_loss = ce_loss + brier_weight * brier_loss
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_ce += ce_loss.item()
        epoch_brier += brier_loss.item()
        
    avg_loss = epoch_loss / len(training_data)
    losses.append(avg_loss)
    if epoch % 3 == 0 or epoch == epochs:
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Total Loss: {avg_loss:.4f} | CE: {epoch_ce/len(training_data):.4f} | Brier: {epoch_brier/len(training_data):.4f}")

# 3. Validation after fine-tuning
decider.readout.eval()
decider.readout.requires_grad_(False)
print("\nRunning post-finetuning evaluation...")
post_scores = []
for item in val_data:
    resp = decider.decide({"state": item["state"], "questions": question})
    choice = resp["answers"]["triage"]["choice"]
    prob = resp["answers"]["triage"]["probabilities"][choice]
    post_scores.append({"state": item["state"][:60], "expected": item["expected"], "pred": choice, "prob": round(prob, 4)})
    print(f"  [Post-FT]  Expected: {item['expected']:<14} | Predicted: {choice:<14} | Prob: {prob:.4f}")

# 4. Save fine-tuned decision head
output_path = Path("finetuned_readout.safetensors")
from safetensors.torch import save_file
save_file({"weight": decider.readout.weight.detach().cpu().contiguous()}, str(output_path))
print(f"\nSuccessfully saved fine-tuned decision head to {output_path} ({output_path.stat().st_size} bytes)")

summary = {
    "epochs": epochs,
    "initial_loss": round(losses[0], 4),
    "final_loss": round(losses[-1], 4),
    "baseline_validation": baseline_scores,
    "post_validation": post_scores
}
with open("finetune_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary to finetune_results.json")
