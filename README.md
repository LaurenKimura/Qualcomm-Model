# The Minimalist Edge AI Challenge, explained from zero

This folder is a project for a one-day student contest. You are given a program that looks at a satellite photo and says what kind of land it is: forest, river, crops, houses, and so on. The program already works. Your score comes from making it smaller, faster, and less power-hungry without making it bad at naming the land.

You do not need an AI class to follow what is in here. Read this page once, then use the commands at the bottom.

## The picture quiz

There are 27,000 satellite photos in a public set called EuroSAT. Each photo is tiny, 64 by 64 pixels, and each one has one of 10 labels.

The contest stretches every photo up to 224 by 224 before the program sees it. That stretch does not add real detail. It is like blowing up a postage stamp until it fills a sheet of paper. The paper is bigger. The information is the same. That fact is the main opening this project uses.

## What the "model" is

A model is a long list of numbers, plus a fixed recipe for combining them.

You hand it a photo. It multiplies and adds those numbers, layer after layer, and finishes with 10 scores, one per land type. The biggest score wins.

The numbers are called **parameters**. The starting model is a small vision network called MobileNetV2. A bigger network, ResNet-50, already taught it. That teaching step is called **distillation**, and it is already done. You are not asked to build a model. You are asked to put the finished one on a diet.

## The four things the judges score

Think of a phone that has to do this quiz all day on a battery.

| What they call it | Plain meaning | Points |
|---|---|---|
| Accuracy and reliability | How often the name is right, including the land type it is worst at | 25 |
| Compression | Fewer stored numbers, and a smaller file | 20 |
| Resource efficiency | Less memory, and less waiting, than the model you started with | 20 |
| Energy parsimony | Less energy per photo than the model you started with | 20 |
| Documentation | A judge can rerun your steps without asking you questions | 10 |
| Engineering judgment | You show the options you tried and why you picked one | 5 |

Speed and energy are scored as a ratio against your own starting measurement, on your own laptop. A faster laptop must not win by itself.

One rule decides most contests: do not buy a tiny file by forgetting a whole land type. Keeping about 97% of the starting accuracy is the top accuracy band in the public scoring sheet.

## The three diets, in the order this project tries them

**1. Shrink the photo on the way in.**

The file still accepts a 224 by 224 photo, because the judges require that. The first step inside the file can shrink it to 160, 128, or 96 before the real work. Less pixels means less arithmetic. The script tries several sizes and writes down the accuracy of each.

**2. Delete weak channels.**

A channel is one pattern-detector inside a layer, like one person on an assembly line. Some of them barely change the answer. Structured pruning removes those workers and closes their stations, so the file really gets smaller.

The obvious PyTorch pruning tool does not do that. It writes zeros in the same stations. The file stays the same size and the wait stays the same. The contest packet warns about this. This project uses `torch-pruning`, which removes the stations.

The first layer and the final "10 answers" layer are left alone. Damaging the first layer blurs everything. Damaging the last layer deletes a land type.

**3. Store each number in 8 bits instead of 32.**

A normal weight is a precise decimal. Eight bits is a coarse integer. The file gets about four times smaller. This is **static quantization**. Beforehand, the script looks at real training photos and picks the coarse scale so the answers barely move. The file format is called QDQ, which is the one Qualcomm's tools accept.

A different shortcut, dynamic quantization, is on purpose not used. The contest notes say the device compiler often rejects it, and it can be slower than the original.

After pruning, the script can do a short repair lesson, a few dozen updates, so the remaining numbers adjust. A long lesson will not finish on a CPU in one day. An unfinished clever idea scores below a finished ordinary one. That sentence is in the contest packet, and the step count is kept small because of it.

## What happens when you run the script

1. It loads the photos and the starting network.
2. It measures the starting network: right answers, worst land type, number of parameters, file size, arithmetic, wait, memory, and energy.
3. It tries the photo sizes and the channel cuts on a sample of the validation photos, and writes one row per try. That table is the tradeoff curve.
4. It rebuilds the best few as ONNX files. ONNX is a portable save format. The judges run the ONNX file, not the Python training code. It also builds an 8-bit copy of each.
5. It keeps the option that still names the photos well and shrinks the cost the most. If nothing holds 97% of the starting accuracy, it says so and still submits the best remaining option.
6. It writes a folder called `submission` with the model, `metrics.json`, a short report, the curve, and a plot.

## What I could not do yet

The contest gives out the starting network, the official photo split, and a baseline score sheet on the morning of the event. Those files are not in this folder. Any accuracy number produced before they arrive is practice, not a score.

The plumbing has a separate test that does not need those files. It builds a blank network of the same shape, deletes 30% of the channels, saves an 8-bit file, and checks that both the parameter count and the file got smaller and that a 224 by 224 photo still produces 10 scores.

## Commands

Open PowerShell in this folder.

Practice check, no contest files needed:

```powershell
.\edge-ai-env\Scripts\python.exe run_pipeline.py --self-test
```

Contest morning, after you have copied in `student_mobilenetv2_distilled.pt`, `val_split.json`, and the `data` folder:

```powershell
.\edge-ai-env\Scripts\python.exe run_pipeline.py --team-name "Your Team"
```

If the laptop is low on memory, add `--batch-size 8`.

Optional, slower, only if the teacher file is present and you want a measured comparison:

```powershell
.\edge-ai-env\Scripts\python.exe run_pipeline.py --team-name "Your Team" --use-teacher
```

Qualcomm device profile, after `qai-hub configure --api_token YOUR_TOKEN` and after the submission file exists:

```powershell
.\edge-ai-env\Scripts\python.exe profile_hub.py
.\edge-ai-env\Scripts\python.exe profile_hub.py --device "THE NAME YOU PICKED"
```

If AI Hub is down, submit the local numbers anyway and say the service was unreachable. The packet says there is no penalty for an outage.

## Files

| File | What it is |
|---|---|
| `run_pipeline.py` | The command you run |
| `edge_ai/data.py` | Loads photos and the official split |
| `edge_ai/compress.py` | Shrink, prune, repair, save, convert to 8-bit |
| `edge_ai/measure.py` | The before-and-after measurements |
| `edge_ai/score.py` | How a candidate is chosen |
| `edge_ai/pipeline.py` | The six steps above |
| `edge_ai/report.py` | The 1-2 page writeup |
| `profile_hub.py` | The Qualcomm device measurement |
| `model1.py` | An old ONNX example. The contest does not use it |
