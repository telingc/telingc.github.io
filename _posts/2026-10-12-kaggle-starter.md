---
layout: post
title: "Kaggle: A Beginner Losing His Mind on ML's Steep Mountains."
excerpt: "Notes from my first Kaggle competition: MRI data, weak labels, experiments, and the frustration of trying to improve a score."
date: 2026-10-12
comments: true
mathjax: true
---


<div class="imgcap">
  <img src="/assets/kaggle-starters/The_Kibo_and_Mawenzi_Cones_of_Mt._Kilimanjaro.png" width="90%" style="display:block; margin:auto;">
  <div class="thecap">Figure 1: Kibo (left) and Mawenzi (right), two peaks of Kilimanjaro. (Source: Wikipedia)</div>
</div>

### Overlook

As a complete newcomer to Kaggle, I used to dream about gold medals the way young climbers dream about summits — full of ambition and a little too much imagination.

Writing this post, though, I was in the middle of the hardest stretch of the competition, feeling properly discouraged. Kaggle, it turns out, is a lot like Kilimanjaro: the altitude is real, and respect comes before glory.

If you ever find yourself in the same place, I hope these notes help. Exploring is just the first step. Let’s get started. ⛺🗻🌌

### Base Camp

Welcome to base camp 🏕️! I hope you're adjusting to the altitude.

#### Preparations

Before jumping into a competition, it helps to have some basic ideas about data science — or at least how neural networks are trained.

Most people submit their work through *Notebooks*. [Google Colab](https://colab.research.google.com/) and [Kaggle Notebooks](https://www.kaggle.com/code) are both convenient and powerful. [Jupyter](https://jupyter.org/) still has its own charm. Any of them is a solid choice.  

One practical tip: use a [virtual Python environment](https://docs.python.org/3/library/venv.html), commonly stored in a `.venv` directory, to keep the project's packages separate.

### Trailhead

If you're still hesitating, honestly, I was moving a bit too fast for my own level. I wasn't very confident, but carrying that fear with me turned out to be useful. Being a little afraid is a good thing: it makes you respect the mountain. Running up a small hill is easy; facing Kilimanjaro with shaky legs is a real adventure.

#### A General Roadmap: The Overview Page

Ahead, Kilimanjaro is waiting. I took [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) as my first challenge - now you know why I was moving a bit too fast. 

Choosing the *Machame* route just for the view can take your breath away in more ways than one.

<div class="imgcap">
  <img src="/assets/kaggle-starters/Mount_Kilimanjaro_Climbing_Routes_and_Huts_photomap-fr.svg.webp" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 2: A map of climbing routes on Kilimanjaro. (Source: Wikipedia)</div>
</div>

Before you start, it helps to have a rough roadmap in mind: leave base camp, reach the trailhead (Overview and first submission), pass a few camps for supplies, face the scree slope (the long stretch of experiments), and finally push for the summit. The **Overview** page is that abstract roadmap. It looks simple, but it matters.

Three parts of the Overview are worth taking seriously:

1. **Description** and **Evaluation** come first. They answer two critical questions: What is the goal? How will your work be scored?
2. **Timeline**. A practical tip is to prefer newer competitions — sometimes this is also a game of time.
3. **Code Requirements**. Saving a new version and waiting for it to run can take hours. Make sure you satisfy every requirement. For example, I once created a notebook and let it run for three hours, only to realize I had not disabled internet access before submitting. A pure waste of time.

#### Imagine the Summit: First Submission

You may already have seen the peak of Kilimanjaro — in videos, photos, or even in dreams. Bringing that same kind of imagination to Kaggle is much easier:

1. Find a public notebook (or code) from the *Code Page*.
2. Hit the *Copy & Edit* button.
3. Save and run the notebook, then wait for it to finish.
4. Submit.

A first **submit** is not the **summit** — just proof that you are on the mountain.

### The Approach March

Kilimanjaro becomes real the moment you step onto the trail. Talking about the mountain is one thing; walking it is another.

For this project — [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — the task is to predict twelve clinically important findings from knee MRI studies. If you have seen [CS231n](https://cs231n.stanford.edu/), CIFAR-10 offers a familiar image-classification example. Here, however, several findings can coexist in one study: this is a **multi-label** task, with a separate prediction for each finding.

#### Longitude, Latitude and Altitude: Data

<div class="imgcap">
  <img src="/assets/kaggle-starters/rsna-knee.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 3: Competition header on Kaggle — a knee MRI from the challenge page.</div>
</div>

I used to think Kaggle was simply about training a well-fitting model. Mountains are not just rock faces to climb — they are about reading the terrain, talking with your team, and studying the routes of those who went before. Kaggle is much the same: a large part of the work is understanding and processing the data.

Before you touch a model, open the **Data** tab. Learn the files, the labels, the missing values, and how a single study is organized. For RSNA that means studies, series, and slices. That map is your longitude, latitude, and altitude.

So what do we actually have? The answers live on the [Data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

##### 1) `train.csv`

This is the main table: **one row per study**. Every row includes a radiology report, but only **58 of 4,407** rows also carry the 12 binary abnormality labels. Those scarce labeled rows are our *golden labels*. The rest — report present, condition labels missing — is one of the central difficulties of this competition.

<table>
  <tr>
    <th>StudyInstanceUID</th>
    <th>Report</th>
    <th>ACL</th>
    <th>MCL</th>
    <th>…</th>
    <th>Effusion</th>
  </tr>
  <tr>
    <td>1.2.826…269</td>
    <td>The study reveals…</td>
    <td>0</td>
    <td>0</td>
    <td>…</td>
    <td>0</td>
  </tr>
  <tr>
    <td>1.2.826…292</td>
    <td>Diz eklemi içi…</td>
    <td></td>
    <td></td>
    <td>…</td>
    <td></td>
  </tr>
  <caption>Top: golden labels (report + 0/1 targets). Bottom: report only, labels missing.</caption>
</table>

Here `0` / `1` mean absence / presence of that finding **when the label exists** — not “this row was diagnosed as abnormal” in general.

##### 2) `train_series.csv`

Each study contains several **series**. A *study* is a single MRI examination; a *series* is one acquisition within that exam — often a different anatomical plane (sagittal, coronal, or axial) or contrast weighting. A patient may have more than one study.

Example rows:

<table>
  <tr>
    <th>StudyInstanceUID</th>
    <th>SeriesInstanceUID</th>
    <th>Fluid_Sensitive</th>
    <th>Fat_Suppression</th>
    <th>Anatomical Plane</th>
  </tr>
  <tr>
    <td>1.2.826…7260</td>
    <td>1.2.826…3772</td>
    <td>1</td>
    <td>1</td>
    <td>Sagittal</td>
  </tr>
  <tr>
    <td>1.2.826…7260</td>
    <td>1.2.826…3265</td>
    <td>1</td>
    <td>1</td>
    <td>Axial</td>
  </tr>
  <tr>
    <td>1.2.826…7260</td>
    <td>1.2.826…6174</td>
    <td>0</td>
    <td>0</td>
    <td>Coronal</td>
  </tr>
  <tr>
    <td>1.2.826…7260</td>
    <td>1.2.826…6697</td>
    <td>1</td>
    <td>1</td>
    <td>Coronal</td>
  </tr>
  <tr>
    <td>1.2.826…7260</td>
    <td>1.2.826…3546</td>
    <td>0</td>
    <td>0</td>
    <td>Sagittal</td>
  </tr>
  <caption>Several series belonging to the same study. Figure 4 shows one slice from each of five series in this study.</caption>
</table>

Simply staring at tables can feel naive, but it often helps. In the example above, the first column stays the same while the second changes: one study, many series. The natural hierarchy is:

$$\text{Patient} \Rightarrow \text{Study} \Rightarrow \text{Series} \Rightarrow \text{Slice}$$

A patient spends some time in the scanner; the result is a study. That study holds multiple series; each series is a stack of slices you can open and inspect.

##### 3) Series and slices

> Code for this section is available [here](https://github.com/telingc/telingc.github.io/tree/main/scripts/post_2/).

Tables are not enough — it helps to look at the images. MRI slices are stored as DICOMs, which are richer and more awkward than ordinary PNGs or JPEGs. Opening a few files with `pydicom` is part of *finding information in the data*: what you can see shapes how you will later batch, sample, and train. Fitting a strong model is only one possible path; reading the data carefully is another, and it comes first.

To make the idea of a series concrete, we take the study above and show **one slice from each of five series**:

<div class="imgcap">
  <img src="/assets/kaggle-starters/montage_planes.png" width="90%" style="display:block; margin:auto;">
  <div class="thecap">Figure 4: Slices from five series of one study. Corner tags: plane and series id. (Source: competition data)</div>
</div>

In this example the study has two sagittal series, two coronal series, and one axial series. Do all studies follow the same pattern? **No.** In `train_series.csv` there are 24,371 series rows and 4,407 studies, so about \(24{,}371 / 4{,}407 \approx 5.53\) series per study on average. Other studies contain more or fewer series, including repeated acquisitions in the same plane.

For slices, we take **22 consecutive slices** from the first series in the table. This is what we get:

<div class="imgcap">
  <img src="/assets/kaggle-starters/grid_22.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">Figure 5: Slices from a single series of one study. (Source: competition data)</div>
</div>

Along one plane (here, sagittal), the scanner steps through the knee and reveals different structures slice by slice. This series happens to have 22 slices. Others may have 34, or 16 — the count is not fixed. Listing a folder does not guarantee anatomical slice order. For the training pipeline, I used image-position and orientation metadata where available to reconstruct the order; `InstanceNumber` was a fallback.

##### I Need to Take a Breath

A short pause on the approach march:

1. **Out of order / irregular shape.** The number of series per study is not fixed, nor is the number of slices per series. Slice order on disk is not reliable until you sort by metadata.
2. **Dirty and incomplete supervision.** Reports are multilingual; precise 0/1 labels exist for only a small subset of studies; the dataset is large.

Different planes and weightings highlight different structures. Before you design a model, it helps to *see* how messy and how rich a single study can be — that is the terrain you are about to climb. This is Kilimanjaro, not a random climbing gym.

This part once drove me crazy. During breaks, I found useful conversations in the competition's Discussion area — the campfire of this particular climb.

#### Campfire Chats: Discussion

At camp, climbers trade weather reports and warn each other about loose rock. On Kaggle, that happens in **Discussion**.

<div class="imgcap">
  <img src="/assets/kaggle-starters/hikers.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 6: Hikers on Kilimanjaro. (Source: Wikipedia)</div>
</div>

Read a few early threads before you invent a clever idea. People will already have asked about label quality, metric quirks, and leakage. You do not have to agree with every comment — but you should know what the mountain has already told others.

For this competition, three themes mattered most to me:

1. **Only 58 golden labels.** I saw this in Discussion before I finished digging through the CSVs myself. It gave me a concrete question to work on: how could I use the reports when most targets were missing?
2. **Baseline routes.** Strong competitors post notebooks, scores, and short write-ups. A public pipeline that already makes a submission is a fixed rope: use it to get on the wall, then replace sections with your own gear.
3. **Shared bottlenecks.** Sooner or later you hit the same walls — sparse labels, messy series, slow DICOM I/O. Discussion is often where those walls are named out loud, sometimes with a partial way around them.

Starting from scratch is a choice, not a duty. On a mountain this size, fixed ropes are there to be used.

#### Fixed Ropes: Validated Notebooks

Some routes are already fixed with ropes. On Kaggle, those are the public notebooks that actually run and score.

Find a baseline that completes end-to-end: load data → train or infer → write `submission.csv`. Copy it, run it, submit once. That first successful pass is not the summit; it is proof that your harness works. After that, you can untie a section of the rope and replace it with your own.

Strong public notebooks often look fast and high-scoring because the hard work is already baked in — processed inputs, saved weights, a working submission path. When the authors “fixed the rope,” they had already tested many holds on that wall and kept only one placement. If you start changing the route, training and debugging will take real time. That is normal. Do not measure your experiments against a one-click inference baseline.

#### Judging the Summit: Evaluation

<div class="imgcap">
  <img src="/assets/kaggle-starters/uhuru_peak_sign.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 7: The sign at Uhuru Peak. (Source: Wikipedia)</div>
</div>

Every mountain has a definition of “the top.” Here it is the **Evaluation** method.

This competition uses **macro-averaged ROC AUC**: calculate an AUC for each of the twelve findings, then take their unweighted mean.

For a smaller example, imagine predicting whether each student has an apple or an orange in their pocket. Both can be present, just as several MRI findings can coexist.

For one fruit, choose a threshold \(t\): predict positive when the model's score is at least \(t\). The **true positive rate** is the fraction of actual positives correctly identified; the **false positive rate** is the fraction of actual negatives incorrectly marked positive. Changing the threshold gives points on the ROC curve, with FPR on the horizontal axis and TPR on the vertical axis.

Here is the example data:

<table>
  <tr>
    <th>Values/Students</th>
    <th>Alice</th>
    <th>Bob</th>
    <th>Charlie</th>
    <th>David</th>
    <th>Eva</th>
  </tr>
  <tr>
    <th>Apple</th>
    <td>True</td>
    <td>True</td>
    <td>False</td>
    <td>True</td>
    <td>False</td>
  </tr>
  <tr>
    <th>Orange</th>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>True</td>
    <td>False</td>
  </tr>
  <tr>
    <th>Apple Prediction</th>
    <td>0.67</td>
    <td>0.78</td>
    <td>0.72</td>
    <td>0.78</td>
    <td>0.34</td>
  </tr>
  <tr>
    <th>Orange Prediction</th>
    <td>0.12</td>
    <td>0.23</td>
    <td>0.34</td>
    <td>0.67</td>
    <td>0.75</td>
  </tr>
  <caption>Two labels and prediction scores for five students. Each student may have either fruit, both, or neither.</caption>
</table>

<div class="imgcap">
  <img src="/assets/kaggle-starters/roc.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">Figure 8: ROC curves for the apple and orange example. Apple AUC = 0.833, orange AUC = 0.750, and macro AUC = 0.792. (Source: Python script)</div>
</div>

The area under each ROC curve summarizes ranking quality. An AUC of 1 means perfect ranking on these examples. For the two fruits, the macro average is approximately \((0.8333 + 0.7500) / 2 = 0.7917\).

The same calculation fits in a few lines of Python:

```python
from sklearn.metrics import roc_auc_score

y_true = [[1, 0], [1, 0], [0, 0], [1, 1], [0, 0]]
y_score = [[.67, .12], [.78, .23], [.72, .34], [.78, .67], [.34, .75]]
print(roc_auc_score(y_true, y_score, average=None))  # [0.8333..., 0.75]
print(roc_auc_score(y_true, y_score, average="macro"))  # 0.7917...
```

See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) for the function and its averaging options.

Knowing how you are scored decides what you optimize: a beautiful model that ignores the metric is still a wrong route.

When Description, Data, Discussion, Notebooks, and Evaluation line up in your head, the approach march is over. The real climbing — experiments, failures, and small gains — starts next.

### The Scree Slopes

My first submission scored **0.891** using the public author's already-trained weights. After switching to models trained in my own experiments, fine-tuning pretrained DINOv2 encoders on the knee data, my best Public LB score was **0.885**. I have recorded a few of those experiments below. The most useful lesson I took from them was:

> **Model performance depends closely on the data pipeline: the labels, the selected views, and the way those views reach the model.**

The experiments made that lesson more concrete.

#### Labels: Building Usable Supervision

The competition provided **58 studies with gold labels**. Only **11 of those were in my usual validation split**; these are different counts. Most training studies had reports but no structured targets. I therefore needed a way to turn the reports into weak supervision. I explored two approaches:

1. **Lexicon and regular-expression extraction:** match findings while handling negation, abbreviations, and variations in wording.
2. **LLM extraction:** request structured predictions, then validate the output and handle uncertain or missing findings.

In my early runs, lexicon-derived labels gave a Public LB score of **0.792**, while a run using the author's LLM-derived table scored **0.824**. I chose LLM extraction as a practical route for this project and later generated my own report-label table. These experiments helped me see how much the supervision rules mattered.

Later attempts to refine or combine the label tables gave mixed results. A table that looked better against the small gold subset did not consistently produce a better image model. I also learned to check whether two runs used the same validation reference and the same submission strategy.

#### Information and Sampling Density: Choosing Model Inputs

Sampling became a practical constraint because I cached resized MRI slices in system RAM. At a higher resolution, each cached slice used more memory, leaving room for fewer slices. I had to balance the number of sampled views, image resolution, and the region of each series that I sampled.

The clearest early result came from increasing the cached slice count from three to six at 224px: the best-single-member submissions scored **0.818** and **0.854**. Other runs explored a larger encoder, overlapping inference windows, and denser central sampling. The table records those changes alongside their individual comparisons.

#### Selected Experiments

<table>
  <tr>
    <th>Advance</th>
    <th>What I used</th>
    <th>Why I tried it</th>
    <th>Observed result</th>
  </tr>
  <tr>
    <td>Report-derived supervision</td>
    <td>LLM-extracted report labels instead of lexicon labels</td>
    <td>Capture language and context that fixed keyword rules miss.</td>
    <td>LB: <strong>0.792 → 0.824</strong></td>
  </tr>
  <tr>
    <td>Member ensembling</td>
    <td>Rank averaging across six models with different seeds and configurations</td>
    <td>Reduce dependence on one model’s errors.</td>
    <td>With my DS labels: <strong>0.819 → 0.837</strong></td>
  </tr>
  <tr>
    <td>More sampled slices</td>
    <td>Increase cached slices per series slot from 3 to 6, at 224px</td>
    <td>Provide more views of each MRI series.</td>
    <td>Best-single-member submissions: <strong>0.818 → 0.854</strong></td>
  </tr>
  <tr>
    <td>Higher resolution</td>
    <td>224px → 336px, alongside a revised member pool</td>
    <td>Preserve finer image detail and improve member composition.</td>
    <td><strong>0.854 → 0.862</strong>; resolution was not isolated.</td>
  </tr>
  <tr>
    <td>Larger encoder</td>
    <td>DINOv2-small → DINOv2-base, with the same 336px/six-slice configuration</td>
    <td>Test whether greater encoder capacity improves task features.</td>
    <td><strong>0.862 → 0.870</strong>, with higher training cost.</td>
  </tr>
  <tr>
    <td>Overlapping-window inference</td>
    <td>Average probabilities over four overlapping windows instead of two non-overlapping windows</td>
    <td>Reduce dependence on a particular grouping of cached slices.</td>
    <td>Same weight package: <strong>0.870 → 0.874</strong></td>
  </tr>
  <tr>
    <td>Denser central sampling</td>
    <td>Narrow the sampling band from 0.06–0.94 to 0.30–0.70, keeping six slices</td>
    <td>Concentrate the fixed sampling budget in the central region.</td>
    <td>Package-inference LB: <strong>0.858 → 0.881</strong></td>
  </tr>
  <tr>
    <td>Complementary supervision blend</td>
    <td>Combine old DS-trained and new-label-trained packages with 75% / 25% weights</td>
    <td>Test whether different supervision produces complementary errors.</td>
    <td>Eight-member LB: <strong>0.885</strong>; internal validation did not establish an improvement.</td>
  </tr>
  <tr>
    <td>Different-fold ensemble</td>
    <td>Four members trained on folds 0, 1, 3 and 4</td>
    <td>Cover all training studies across the ensemble while retaining four members.</td>
    <td>LB: <strong>0.885</strong>, matching the eight-member blend with fewer model forwards.</td>
  </tr>
  <caption>
    Selected experimental advances. Each row describes its own comparison,
    rather than a continuous sequence of otherwise identical models.
    Scores are observed Public LB results; differences alone do not establish repeatable gains.
  </caption>
</table>

### Back at Camp

<div class="imgcap">
  <img src="/assets/kaggle-starters/hiker.png" width="30%" style="display:block; margin:auto;">
  <div class="thecap">Figure 9: Kaggle hiker. (Source: Kaggle start page)</div>
</div>

I dream big but do things less. and I've been into this for weeks having no idea what I'm doing. Losing my mind like the title. I started this competition thinking about medals. Over the following weeks, I learned how much work sits between a folder of MRI files and a prediction I can explain. But the more I read the more I suspect that this isn't the place I want to be.

I also found it demanding to turn every result into a reason for one more experiment. A close friend commented that I'm someone who always preparing for a life but never had lived once. Honestly it's true, I can't stand being a beginner for everything. However, I'm closing this part of the project and giving myself time to digest it. But the mountain is still there, welcoming every ambitious hikers to learn from it.

If you are starting your first Kaggle competition, I hope this account gives you a clearer view of the work involved. I am heading back to camp with a pipeline I understand better and a set of experiments I can explain. Thanks and sorry that I can't give any other takeway tips.