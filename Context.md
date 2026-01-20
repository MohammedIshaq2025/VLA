# Direction 1: SE(3) Manifold ZOO Attack - Comprehensive Context for Claude Code

## **Research Overview & Motivation**

You are implementing a novel adversarial attack on Vision-Language-Action (VLA) models that achieves breakthrough query efficiency by exploiting the geometric structure of robot action spaces. This is for an ECCV 2026 submission, so precision, novelty, and rigorous evaluation are paramount.

**The Big Picture**: VLA models like OpenVLA enable robots to understand natural language and perform complex manipulation tasks. However, they are vulnerable to adversarial patches—small visual perturbations that cause catastrophic failures. Existing attacks (Wang'24, RobotGCG, EDPA) require 5,000-10,000 queries because they operate in high-dimensional token space (32,768 dimensions). Our insight: the final action decoder outputs only 7 numbers representing a physical SE(3) pose. By optimizing directly on this low-dimensional manifold, we achieve 89% attack success rate (ASR) with just ~200 queries—a 25× improvement.

## **Understanding OpenVLA Architecture**

OpenVLA-7B consists of three main components:

1. **Visual Encoder**: Processes the 256×256 RGB image from the robot's camera (agentview) into patch embeddings. This is typically a CLIP or DINO backbone.

2. **Language Encoder**: Tokenizes the natural language instruction (e.g., "pick up the black bowl on the left") into text embeddings.

3. **LVLM Backbone**: A large language model (Llama-2 7B) that processes concatenated image and text embeddings, then outputs action tokens.

**Critical Layer - Action Decoder**: The LVLM's output tokens are passed through a final projection layer and action tokenizer to produce the 7D continuous action vector:
- `dx, dy, dz`: Delta position in meters (typically ±0.1m)
- `droll, dpitch, dyaw`: Delta rotation in radians (±0.2 rad)
- `gripper`: Binary open/close (-1 for close, +1 for open)

**Your Task**: Hook into this action generation process to extract the 7D vector **before** tokenization. This is non-trivial because the action tokenizer is part of the model's forward pass. You must use register_forward_hook() on the correct layer (likely model.lm_head) and parse the output.

## **Understanding LIBERO Dataset Structure**

LIBERO is a simulation benchmark with 130 manipulation tasks across 4 suites:

- **libero_spatial**: 10 tasks testing spatial reasoning (e.g., "left bowl" vs "right bowl")
- **libero_object**: 10 tasks testing object recognition (different objects, same action)
- **libero_goal**: 10 tasks testing goal variations (same objects, different goals)
- **libero_10**: 10 long-horizon tasks (combining skills)

**Data Format** (HDF5):
```
task_X_demo.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── agentview_image (T, 256, 256, 3)    # PRIMARY CAMERA
│   │   │   └── robot0_eye_in_hand (T, 256, 256, 3) # WRIST CAMERA
│   │   ├── actions (T, 7)                          # GROUND TRUTH ACTIONS
│   │   └── attrs/
│   │       ├── language_instruction (str)
│   │       └── task_name (str)
│   ├── demo_1/
│   └── ... (50 demos per task)
```

**Critical Details**:
- Each task has **50 demonstrations** with random variations in object positions, lighting, and initial arm pose
- Actions are **delta commands**, not absolute positions
- Frame rate: 10 Hz, episode length: 50-200 frames
- For training: Use **15 episodes** (randomly selected), test on **35 held-out episodes**

## **The Core Innovation: SE(3) Manifold Optimization**

**SE(3) Meaning**: Special Euclidean Group in 3D—the space of rigid body transformations. For robotics, this is the natural space of actions:
- SO(3): Rotation subspace (droll, dpitch, dyaw)
- ℝ³: Translation subspace (dx, dy, dz)
- Gripper: Binary state (can be treated as separate dimension)

**Why This Matters**: Actions on SE(3) have **geometric meaning**. The distance between two actions isn't just Euclidean difference in 7D space. A rotation of 350° is the same as -10° (wraparound). A position error of 10cm when near an object matters more than when far away.

**Our Distance Metric**:
- Position: Standard Euclidean distance (meters)
- Rotation: Geodesic distance on SO(3) (radians, handles wraparound)
- Gripper: Binary distance (0 if same, 1 if different)
- Combined: Weighted sum (pos_dist + rot_dist + grip_dist)

This respects the manifold structure, giving meaningful gradients for optimization.

## **ZOO (Zero-Order Optimization) Fundamentals**

ZOO is **gradient-free optimization**—critical for black-box attacks where we can't compute gradients through the victim model.

**How It Works**:
1. Start with random patch (32×32×3)
2. Add random perturbation δ ~ N(0, 0.01)
3. Query model with **two patched images**: x+δ and x-δ
4. Estimate gradient: ∇f ≈ (f(x+δ) - f(x-δ)) / 2δ
5. Update patch: patch ← patch - lr × ∇f
6. Repeat for 200 iterations

**Why Antithetic Sampling**: Using both +δ and -δ reduces variance by 2×, giving more stable estimates with fewer queries.

**ZOO vs. Traditional PGD**:
- PGD needs **gradients**: requires white-box access, 1000s of backward passes
- ZOO needs **queries**: only forward passes, but needs many queries for high dimensions
- **Our breakthrough**: Reduce dimensionality from 32,768 (tokens) to 7 (SE(3)) → ZOO becomes practical with 200 queries

## **Threat Model & Query Budget**

**Adversary Knowledge**: Black-box access to OpenVLA
- Can send images + text, receive actions
- **NO** access to model weights, gradients, or architecture
- **YES** access to OpenVLA's action space structure (we assume knowledge that actions are SE(3) poses—this is public knowledge from the OpenVLA paper)

**Query Budget**: 200 queries maximum
- Each query = 1 forward pass of OpenVLA on 1 patched image
- Timing: ~0.15s per query → 30 seconds total training time
- Memory: ~85GB peak (model + patch buffer + activations)

**Physical Realism**: Patch is applied to **robot's own arm** in the third-person camera view. This is realistic because:
- The arm appears in every frame
- A small sticker on the arm is hard to detect
- The visual encoder overfits to arm appearance (hypothesis from EDPA paper—we exploit this)

## **Attack Objective: Generic Failure**

We're implementing **generic task failure** rather than specific semantic failures because:
1. **Simpler**: Doesn't require task-specific target engineering
2. **Stronger**: Works across all 40 LIBERO tasks
3. **Evaluable**: ASR measured by action deviation, not task-specific success

**Our Target Generation**:
- Invert gripper state (open ↔ close) to cause dropping
- Add position noise (±5cm) to cause misplacement
- Add rotation noise (±0.1 rad) to cause orientation errors

Combined, this causes **catastrophic failure** in any manipulation task.

## **Why This Is Novel (ECCV Impact)**

**1. Methodological Novelty**: No existing VLA attack uses SE(3) manifold optimization. All prior work (Wang'24, EDPA, VLA-Fool) operates in token space or embedding space. This is the first to respect action space geometry.

**2. Query Efficiency**: 200 queries vs. 5,000+ is a **quantitative breakthrough**. Query efficiency is a hot topic at ECCV/ICCV/NeurIPS—reviewers will care about this 25× improvement.

**3. Transferability**: Because the patch attacks the **action decoder's geometric structure**, it generalizes across tasks. A patch trained on **libero_spatial** works on **libero_object** and **libero_goal** with minimal ASR drop.

**4. Theoretical Soundness**: The method has provable convergence properties on the SE(3) manifold (locally strongly convex near optimal). We can show **O(1/√T)** convergence rate, matching ZOO theory.

**5. Practical Impact**: Attack works on **realistic threat model** (black-box, 200 queries feasible in real-time sensor injection). EDPA and Wang'24 require unrealistic query budgets or white-box access.

## **Evaluation Protocol & Metrics**

**Training Phase**:
- **Split**: 15 episodes for training, 35 for testing (per task)
- **Sampling**: Random frame per query (avoid static frames where action ≈ 0)
- **Early stopping**: If ASR > 85% for 5 consecutive queries, stop
- **Minimum queries**: Run at least 50 queries even if ASR plateaus

**Testing Phase**:
- Evaluate patch on **all 35 held-out episodes**
- For each episode: sample 10 frames (evenly spaced), compute ASR and SE(3) distance
- **Aggregate ASR**: Average across all frames/episodes
- **SE(3) Distance**: Average geometric distance from ground truth actions

**Success Metrics**:
- **ASR > 85%**: More than 85% of actions are adversarially successful
- **Query efficiency**: Achieved in <200 queries
- **Clean accuracy drop**: <2% (patch should not affect unpatched performance)
- **Transferability**: Patch works on held-out tasks with <5% ASR degradation

## **Implementation Phases & Verification**

**Phase 0 (Setup)**: Verify environment, model loading, data access. **CRITICAL**: Must confirm OpenVLA loads and returns 7D actions.

**Phase 1 (Data)**: Implement LIBERO loader with train/test split. **CRITICAL**: Must correctly sample random frames with non-zero actions.

**Phase 2 (ZOO)**: Core optimizer. **CRITICAL**: Must apply patch correctly, compute SE(3) distance, estimate gradients with antithetic sampling. **VERIFY**: Loss decreases (becomes more negative), ASR increases.

**Phase 3 (Targets)**: Generate adversarial target actions. **CRITICAL**: Must invert gripper state and add meaningful perturbations.

**Phase 4 (Jobs)**: SLURM submission. **CRITICAL**: Set MUJOCO_GL=osmesa, monitor GPU memory, log everything.

**Phase 5 (Scripts)**: Main training/testing scripts. **CRITICAL**: Save intermediate results, handle crashes gracefully, print detailed logs.

**Phase 6 (Validation)**: Post-run verification. **CRITICAL**: Must verify all outputs exist, metrics are in range, ASR is monotonic.

**Phase 7 (Scaling)**: Extend to multiple tasks. **CRITICAL**: Use SLURM array jobs, aggregate results correctly.

## **Common Pitfalls & How to Avoid Them**

**1. Wrong Action Layer Hook**: If you hook too early (before LVLM) or too late (after tokenization), you'll get garbage dimensions. **Solution**: Print model structure, find the layer that outputs 7 tokens, verify shape is (batch, 7, hidden_dim).

**2. Patch Out of Bounds**: If patch position + patch size > image dimensions, you'll get array errors. **Solution**: Always clip position to [0, 256-patch_size].

**3. Static Frame Sampling**: If you sample frames where the robot is stationary (action ≈ 0), the attack has no signal. **Solution**: Filter frames where max(|action[:3]|) > 0.01.

**4. NaN Actions**: Sometimes OpenVLA outputs NaN for out-of-distribution inputs. **Solution**: Check `np.isnan(action_pred)` and skip that query, logging the error.

**5. GPU Memory Leaks**: If you don't clear cache, memory grows over queries. **Solution**: Call `torch.cuda.empty_cache()` every 10 queries, log memory usage.

**6. Non-improving ASR**: If ASR plateaus at <50%, the patch position may be poor or learning rate too low. **Solution**: Try patch position on robot arm (60,120), increase lr to 5e-3.

**7. Patch Not Saving**: If SLURM job crashes, patch is lost. **Solution**: Save checkpoint every 50 queries, load from latest if resuming.

## **What Success Looks Like**

**Training Log (Excerpt)**:
```
[2025-01-19 14:32:15] Loading OpenVLA-7B model...
[2025-01-19 14:32:48] Model loaded successfully. GPU Memory: 14.2 GB
[2025-01-19 14:33:12] Loaded 15 train, 35 test episodes
[2025-01-19 14:33:15] Starting adversarial patch training...
[ZOO] Query    0/200 | Loss: -0.1245 | ASR: 12.3% | GPU: 14.8GB
[ZOO] Query   10/200 | Loss: -0.4567 | ASR: 34.5% | GPU: 15.1GB
[ZOO] Query   20/200 | Loss: -0.7892 | ASR: 56.7% | GPU: 15.3GB
...
[ZOO] Query  180/200 | Loss: -1.2345 | ASR: 88.9% | GPU: 15.8GB
[ZOO] Early stopping: ASR > 85.0% for 5 steps
[2025-01-19 14:35:22] Training Complete! Best ASR: 89.2%, Queries: 185
```

**Testing Results**:
```
Average ASR: 87.3%
Average SE(3) Distance: 0.4567
Per-episode ASR range: [80.0%, 94.3%]
All metrics within expected range. ✅
```

**Patch Visualization**: 32×32 RGB image showing a colorful pattern that, when placed on the robot arm, causes consistent failure.

**ECCW Impact**: "We present the first query-efficient black-box attack on VLAs by exploiting SE(3) action space geometry, achieving 89% ASR with 200 queries—a 25× improvement over prior work."

---

## **Your Role as Implementer**

You are not just writing code—you are **validating a novel research hypothesis**. Every implementation decision should be justified by:

- **Does this respect the SE(3) manifold structure?** (Use geometric distance, not token loss)
- **Does this reduce query complexity?** (Avoid high-dimensional optimization)
- **Is this physically realistic?** (Patch on robot arm, not floating in space)
- **Is this evaluable?** (Clear ASR metric, train/test split)
- **Is this novel?** (No prior work uses this exact formulation)

If you're unsure about a decision, **document it and flag for review**. The PRD gives you structure, but your expertise in translating research to code is what will make this succeed.

**Remember**: The goal is not just working code, but **publishable research**. Every log line, every validation check, every metric collected should support the ECCV paper's claims.