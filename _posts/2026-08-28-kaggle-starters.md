---
layout: post
title: "Kaggle: The Starters' Trekking Guide to ML's Steep Mountains."
excerpt: "Trying my best in a Kaggle competition — a pure mental test. Hope this can be a roadmap if you'd like to know a bit about Kaggle."
date: 2026-08-28
comments: true
mathjax: true
---


<div class="imgcap">
  <img src="/assets/kaggle-starters/The_Kibo_and_Mawenzi_Cones_of_Mt._Kilimanjaro.png" width="90%" style="display:block; margin:auto;">
  <div class="thecap">Figure 1: Kilimanjaro's main peaks are Kibo (left) and Mawenzi (right), as named by the Chagga. They mean "spotted" and "broken" respectively. (Source: Wikipedia)</div>
</div>

### Overlook

As a complete newcomer to Kaggle, I used to dream about gold medals the way young climbers dream about summits — full of ambition and a little too much imagination.

Writing this post, though, I was in the middle of the hardest stretch of the competition, feeling properly discouraged. Kaggle, it turns out, is a lot like Kilimanjaro: the altitude is real, and respect comes before glory.

If you ever find yourself in the same place, I hope these notes help. Exploring is just the first step. Let’s get started. ⛺🗻🌌

> *"When altitude tempts your mind to surrender, victory belongs to the individual who can focus entirely on the single next step."*
> — Henry Stedman, *Kilimanjaro: A Trekking Guide to Africa's Highest Mountain*

### Base Camp

Warm welcome from base camp 🏕️! Hope you're adjusting well to the altitude.

#### Preparations

Before jumping into a competition, it helps to have some basic ideas about data science — or at least how neural networks are trained.

Most people submit their work through *Notebooks*. [Google Colab](https://colab.research.google.com/) and [Kaggle Workspace](https://www.kaggle.com/work) are both convenient and powerful. [Jupyter](https://jupyter.org/) still has its own charm. Any of them is a solid choice.  

One practical tip: using a virtual Python environment([.venv](https://docs.python.org/3/library/venv.html)) helps a lot, especially when you need to install many packages for a specific project.

### Trailhead

If you're still hesitating, honestly, I was moving a bit too fast for my own level. I wasn't very confident, but carrying that fear with me turned out to be useful. Being a little afraid is a good thing: it makes you respect the mountain. Running up a small hill is easy; facing Kilimanjaro with shaky legs is a real adventure.

#### A General Roadmap: The Overview Page

Ahead, Kilimanjaro is waiting. I took [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) as my first challenge - now you know why I was moving a bit too fast. 

Choosing the *Machame* route just for the view can take your breath away in more ways than one.

<div class="imgcap">
  <img src="/assets/kaggle-starters/Mount_Kilimanjaro_Climbing_Routes_and_Huts_photomap-fr.svg.webp" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 2: Machame(the brown marked route), Second most popular route, incredible view but a bit challenging.(Source: Wikipedia)</div>
</div>

Before you start, it helps to have a rough roadmap in mind: leave base camp, reach the trailhead (Overview and first submission), pass a few camps for supplies, face the scree slope (the long stretch of experiments), and finally push for the summit. The **Overview** page is that abstract roadmap. It looks simple, but it matters.

Three parts of the Overview are worth taking seriously:

1. **Description** and **Evaluation** come first. They answer two critical questions: What is the goal? How will your work be scored?
2. **Timeline**. A practical tip is to prefer newer competitions — sometimes this is also a game of time.
3. **Code** **Requirements**. Saving a new version and waiting for it to run can take hours. Make sure you satisfy every requirement. For example, I once created a notebook and let it run for three hours, only to realize I had not disabled internet access before submitting. A pure waste of time.

#### Imagine the Summit: First Submission

You may already have seen the peak of Kilimanjaro — in videos, photos, or even in dreams. Bringing that same kind of imagination to Kaggle is much easier:

1. Find a public notebook (or code) from the *Code Page*.
2. Hit the *Copy & Edit* button.
3. Save and run the notebook, then wait for it to finish.
4. Submit.

A first **submit** is not the **summit** — just proof that you are on the mountain.

### The Approach March

Kilimanjaro becomes real the moment you step onto the trail. Talking about the mountain is one thing; walking it is another.

For this project — [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — the task is to classify knee MRI studies into 12 targets: twelve clinically important abnormalities. If you have seen [CS231n](https://cs231n.stanford.edu/), CIFAR-10 is a simpler cousin of the same idea: teach a machine to recognize ships, planes, birds, and more — in textbook terms, 50,000 images into 10 classes.

#### Longitude, Latitude and Altitude: Data

<div class="imgcap">
  <img src="/assets/kaggle-starters/rsna-knee.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 3: Competition header on Kaggle — a knee MRI from the challenge page.</div>
</div>

I used to think Kaggle was simply about training a well-fitting model. Mountains are not just rock faces to climb — they are about reading the terrain, talking with your team, and studying the routes of those who went before. Kaggle is much the same: a large part of the work is understanding and processing the data.

Before you touch a model, open the **Data** tab. Learn the files, the labels, the missing values, and how a single study is organized. For RSNA that means studies, series, and slices. That map is your longitude, latitude, and altitude.

So what do we actually have? The answers live on the [Data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

##### 1) `train.csv`

This is the main table: **one row per study**. Every row includes a radiology report, but only about **58 of 4,407** rows also carry the 12 binary abnormality labels. Those scarce labeled rows are our *golden labels*. The rest — report present, condition labels missing — is one of the central difficulties of this competition.

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

Each study contains several **series**. A *study* is one exam for one patient; a *series* is one acquisition within that exam — often a different anatomical plane (Sagittal, Coronal, Axial) or contrast weighting.

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
  <caption>Several series belonging to the same study. Figure 5 shows one slice from each of five series in this study.</caption>
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

In this example the study has two sagittal series, two coronal series, and one axial series. Do all studies follow the same pattern? **No.** In `train_series.csv` there are 24,371 series rows and 4,407 studies, so about \($24371 / 4407 \approx 5.53$\) series per study on average. Some exams look like “3 sagittal + 3 axial + 4 coronal,” which is nothing like the example above.

For slices, we take **22 consecutive slices** from the first series in the table. This is what we get:

<div class="imgcap">
  <img src="/assets/kaggle-starters/grid_22.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">Figure 5: Slices from a single series of one study. (Source: competition data)</div>
</div>

Along one plane (here, sagittal), the scanner steps through the knee and reveals different structures slice by slice. This series happens to have 22 slices. Others may have 34, or 16 — the count is not fixed. Slice files are also not guaranteed to appear in anatomical order when you list a folder; you typically sort them by `InstanceNumber` (or an equivalent field) in the DICOM metadata.

##### I Need to Take a Breath

A short pause on the approach march:

1. **Out of order / irregular shape.** The number of series per study is not fixed, nor is the number of slices per series. Slice order on disk is not reliable until you sort by metadata.
2. **Dirty and incomplete supervision.** Reports are multilingual; precise 0/1 labels exist for only a small subset of studies; the dataset is large.

Different planes and weightings highlight different structures. Before you design a model, it helps to *see* how messy and how rich a single study can be — that is the terrain you are about to climb. This is Kilimanjaro, not a random climbing gym.

This part once drove me crazy. Luckily I'm not alone on the march, when it comes to rest times, there's always chats around campfires you can join.

#### Campfire Chats: Discussion

At camp, climbers trade weather reports and warn each other about loose rock. On Kaggle, that happens in **Discussion**.

<div class="imgcap">
  <img src="/assets/kaggle-starters/hikers.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Figure 6: Hikers on Kilimanjaro. (Source: Wikipedia)</div>
</div>

Read a few early threads before you invent a clever idea. People will already have asked about label quality, metric quirks, and leakage. You do not have to agree with every comment — but you should know what the mountain has already told others.

For this competition, three themes mattered most to me:

1. **Only 58 golden labels.** I saw this in Discussion before I finished digging through the CSVs myself. It told me where to focus: that is what do we do with reports and missing targets?
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
  <div class="thecap">Figure 7: Sign of Uhuru peak.(Source: Wikipedia)</div>
</div>

Every mountain has a definition of “the top.” Here it is the **Evaluation** metric.

Read it carefully. For this competition it is a macro-averaged AUC over 12 findings — so each abnormality matters, not only the easy ones. Knowing how you are scored decides what you optimize: a beautiful model that ignores the metric is still a wrong route.

When Description, Data, Discussion, Notebooks, and Evaluation line up in your head, the approach march is over. The real climbing — experiments, failures, and small gains — starts next.