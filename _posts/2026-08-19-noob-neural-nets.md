---
layout: post
title: "Visualizing a simple neural network under the hood, from a naive view."
excerpt: "Visualizing activation, gradients & batch normalization — a brief demonstration of neural nets built the old-fashioned way, from a NOOB perspective. A short hike through nn-zero-to-hero."
date: 2026-08-19
comments: true
mathjax: true
---

[Makemore](https://github.com/karpathy/makemore) by [Andrej](https://karpathy.ai/) confused me quite a lot while I was learning part 3. If topics like BatchNorm and Kaiming Init confuse you as much as they confused me, I hope you find this helpful.
The goal is to visualize activations, gradients, and batch normalization — a brief demonstration of neural nets built the old-fashioned way, from a NOOB perspective. Enjoy this short hike through [nn-zero-to-hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).
Let's jump right in.

### WAAAIT, it's just a table of probabilities!

Training a bigram neural net — one that takes the previous two characters, embeds them, passes them through a hidden layer, and then computes a softmax to obtain a probability distribution over the next character — can seem futile. Because the normalized logits look exactly the same as the statistics in the graph below.

That was literally my naive thought after drawing this graph:
> *"How about just looking this table up manually? Like the musing of an HLM (Human Language Model 🧠🤖)"*

<div class="imgcap">
  <img src="/assets/2026-08-19/makemore_contrast.png" width="80%" style="display:block; margin:auto;">
  <div class="thecap">1) Bigram neural net visualization</div>
</div>

Looking at the graph above (darker blocks mean higher probability), it's only natural to wonder: why not build this table from statistics and simply sample from it? What the net ultimately outputs are probabilities that could be obtained by counting anyway.

<table>
  <tr>
    <th>bigram</th>
    <th>eightgram WaveNet</th>
  </tr>
  <tr>
    <td>carpavela</td>
    <td>lovelton</td>
  </tr>
  <tr>
    <td>jhquiffirleigelty</td>
    <td>attherin</td>
  </tr>
  <tr>
    <td>halayan</td>
    <td>laikaela</td>
  </tr>
  <tr>
    <td>jazonte</td>
    <td>amalah</td>
  </tr>
  <tr>
    <td>den</td>
    <td>auri</td>
  </tr>
  <tr>
    <td>rha</td>
    <td>cayleigh</td>
  </tr>
  <tr>
    <td>kaeli</td>
    <td>phood</td>
  </tr>
  <tr>
    <td>nellara</td>
    <td>glensie</td>
  </tr>
  <tr>
    <td>chaiiy</td>
    <td>trulan</td>
  </tr>
  <tr>
    <td>kaleigh</td>
    <td>osway</td>
  </tr>
  <tr>
    <td>ham</td>
    <td>anzey</td>
  </tr>
  <tr>
    <td>join</td>
    <td>ayda</td>
  </tr>
  <tr>
    <td>quinn</td>
    <td>eunas</td>
  </tr>
  <tr>
    <td>shoine</td>
    <td>diona</td>
  </tr>
  <tr>
    <td>livabi</td>
    <td>colsleigh</td>
  </tr>
  <tr>
    <td>wanelo</td>
    <td>suo</td>
  </tr>
  <tr>
    <td>dearynix</td>
    <td>jatteren</td>
  </tr>
  <tr>
    <td>kaelynn</td>
    <td>jerine</td>
  </tr>
  <tr>
    <td>demed</td>
    <td>whytus</td>
  </tr>
  <tr>
    <td>edi</td>
    <td>kaylie</td>
  </tr>
  <caption>name-like comparison</caption>
</table>

But the truth is usually more nuanced. If you want to scale up and get better efficiency and performance — predictions that hold up even with a smaller training set — the extra context offered by a trigram (or longer) model is often the better choice. Looking at the table above, the eightgram WaveNet outputs sound much more *"name-like"* to me. At that point, the probability model looks like this:

<div class="imgcap">
  <img src="/assets/2026-08-19/cube.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">2) Trigram neural net visualization</div>
</div>

The 3D heatmap on the left represents the probability that the next character is *a*, conditioned on different character combinations. For example, the tiny green dot stands for the probability of `emm → a`, which is 28%.

$$P(\text{next character is }a \mid \text{context is }emm) \approx 0.28$$

And there are 27 cubes like this for the simplest net without batchnorm. Even though it's a silly idea, I still think it's beautiful. 🥹
(Drawing this was just fun and cool — ignore me. 😄)

### High initial loss caused by improper initialization

<div class="imgcap">
  <img src="/assets/2026-08-19/lossi.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">3) loss.log10() decreasing with training</div>
</div>

As you can see, the initial loss is $\geqslant 10^{1.4} \approx 25.12$. But we have no reason to believe that any character has a higher chance than any other, so the loss under a uniform distribution is what we'd expect. It can be calculated as:

$$-\ln\frac{1}{27} = 3\ln3 \approx 3.29583687$$

I verified it in IDLE — here's how it goes:

```python
>>> import torch
>>> uni_dis = torch.ones(1, 27) * 1 / 27
>>> loss = -(uni_dis.log()).mean()
>>> loss.item()
3.295837163925171
```

> Playing with this toy model, when Andrej saw this insanely high loss but had to save the punchline for the next class — that must have been challenging. 😂

What's happening is that `torch.randn` generates numbers from a standard normal distribution, $out_i \sim \mathcal{N}(0,1)$. So after the hidden layer, tanh, and softmax, the probabilities can hardly be uniform — sadly, they get pretty "paranoid."

By chance, I could get a crazy low loss because my net happens to be confidently correct on a random guess, but more often than not, I'll run into a big, big error. You'll see in the graph below that the distribution of C's values roughly follows a Gaussian, but when you look at the mean prediction across this batch, the chance of getting a *t* in five trials is "promising" (in the wrong way). The yellow line is the uniform distribution; compared to what we actually have, there's a lot of room for optimization.

<div class="imgcap">
  <img src="/assets/2026-08-19/gaussian.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">4) Improper initialization.</div>
</div>

How do we make the `probs` more uniform? The simplest idea is to find a way to reduce the variance of the data — and the object to operate on is, of course, the logits. The hidden layer shouldn't be touched, so scaling down `W2` and initializing `b2` to zero sounds reasonable: since it's a matrix multiplication, scaling down a bit can bring a nice drop in loss.

```python
# ...
g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),                  generator=g)
W1 = torch.randn((block_size * n_embd, n_hidd),        generator=g) # * 0.1
b1 = torch.randn(n_hidd,                               generator=g) # * 0.01
W2 = torch.randn((n_hidd, vocab_size),                 generator=g) * 0.01
b2 = torch.randn(vocab_size,                           generator=g) * 0
# ...
```

The result: the initial loss dropped from 27.9 to 3.3, slightly above $3\ln3$, and the final performance improved by a decent margin (val loss: 2.15 → 2.13). More importantly, efficiency — we now skip the long slog down the steep hockey-stick curve. 🏒

<div class="imgcap">
  <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px;">
    <img src="/assets/2026-08-19/lossi.png" width="45%" style="display:block; margin:auto;">
    <img src="/assets/2026-08-19/lossi-new.png" width="45%" style="display:block; margin:auto;">
  </div>
  <div class="thecap">5) loss.log10() decreasing with training</div>
</div>

### Do neural networks dream of dead neurons?

As Andrej says, there's a problem hiding in this simple net. To spot it, we need one line of code to record our gradients:

```python
# --- minibatch construct --->>>
ix     = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]
# --- forward pass          --->>>
emb     = C[Xb]
embcat  = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1 + b1
h       = torch.tanh(hpreact)
logits  = h @ W2 + b2
loss    = F.cross_entropy(logits, Yb)
# --- backward pass         --->>>
for p in parameters:
    p.grad = None

[t.retain_grad() for t in [hpreact]] # <<<--- THIS IS IT!

loss.backward()
# --- update               --->>>
lr = 0.1
p.data += -lr * p.grad
```

If I just plot the histogram of the output and look at the gradients of the pre-activations, I think this is even more sweat-inducing than watching the video. Just stare at this:

1. $$\frac{\partial}{\partial x} \tanh{x}\bigg|_{x=\text{pre-activation}_{ij}} = 0$$

2. `p.data += -lr * p.grad`

This means that `pre_activation_ij.grad` is 0, and for that neuron, it will never be updated! These dead neurons are just like me in a university lecture — sitting there and learning NOTHING. We have 5,000 more neurons like this hiding behind the tanh layer. It's generous to say our hidden layer is a university.

<div class="imgcap">
  <img src="/assets/2026-08-19/dead_neurons.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">6) Dead neurons.</div>
</div>

Enough backward-passing for now. Let's bring this question forward and figure out where this comes from. Actually, people notice this problem from the activation `h`: there are tons of 1.00s in it. The region around $\pm1$ holds most of the distribution density of values in `h`; if $h_{ij} = 1$, then the corresponding value in `pre-activation.grad` is 0. BUT! you might argue, it can't be *exactly* 1 because the input can never reach infinity — so why is that? Right, and floating-point precision isn't fine enough to guarantee that tanh of very large numbers isn't exactly 1. In the left graph below, the red curve is the gradient while the blue area is the distribution. When values fall outside the orange lines, their gradients tend toward zero, per $\frac{\partial}{\partial x}\tanh{x}$.

<div class="imgcap">
  <img src="/assets/2026-08-19/density.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">7) Values sit in the plateau of tanh with near-zero gradients.</div>
</div>

I hope that's clear enough. Intuitively, the completely white bars are dead neurons. In that case, the net is simply inactive. If the picture looks pale, you should be worried too.

<div class="imgcap">
  <img src="/assets/2026-08-19/whites.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">8) The whiter the picture, the more inefficient the net.</div>
</div>

To fix this, imagine squeezing the pre-activation distribution to make it narrower, so values don't cross the orange line. Scaling down W1 and b1 is a good choice.

```python
# ...
g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),                  generator=g)
W1 = torch.randn((block_size * n_embd, n_hidd),        generator=g) * 0.1
b1 = torch.randn(n_hidd,                               generator=g) * 0.01
W2 = torch.randn((n_hidd, vocab_size),                 generator=g) * 0.01
b2 = torch.randn(vocab_size,                           generator=g) * 0
# ...
```

Now it's much better. The final loss dropped from the original 2.15 to 2.13 earlier, and now to 2.10. And if the distribution gets too concentrated, then tanh is effectively doing nothing.

<div class="imgcap">
  <img src="/assets/2026-08-19/correct_init.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/goodgood_neurons.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">9) Now they're motivated neurons.</div>
</div>

### The myth, the legend — Kaiming Initialization!

```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200) * magical_number
y = x @ w
print(x.mean(), y.mean())
print(x.std(),  y.std())
```

From the code above, what can we expect for the mean and standard deviation?
1. magical number = 1 → standard deviation grows.
2. magical number > 1 → standard deviation grows even more.
3. 0 < magical number < 1 → standard deviation shrinks.

The whole point is to find a way to prevent the normal distribution from collapsing to 0 or blowing up to infinity — to keep the standard deviation at 1. [Kaiming Initialization](https://arxiv.org/abs/1502.01852) suggests the magical number should be $\sqrt{\frac{2}{fan_{in}}}$ for the forward pass, and $\sqrt{\frac{gain^2}{fan_{in}}}$ for the backward pass, where gain depends on the non-linearity. For $\tanh{x}$, $gain = \frac{5}{3}$. They differ because the forward and backward passes contract the distribution in different ways, so different gains are needed to counteract the non-linearity.

<div class="imgcap">
  <img src="/assets/2026-08-19/kaiming.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/kaiming_neurons.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">10) Now they're stable Kaiming neurons; activations are properly initialized.</div>
</div>

Configuring the gain is nontrivial. Choosing $gain = \frac{5}{3}$ for initialization makes sense: compared to $gain = 1$, tanh is a squashing function that pulls the distribution toward 0, so a lower gain would struggle to counteract that tendency.

And for $gain = 4$, the activations tend to be far too saturated.

<div class="imgcap">
  <img src="/assets/2026-08-19/gain=1.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/gain=5over3.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/gain=4.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">11) Different gains.</div>
</div>

Once we add BatchNorm, the activations become much less sensitive to initialization.

> The math behind it — I'd like to leave that for another time.

### Confusing BatchNorm

Precisely tuned, fragile initializations matter less today thanks to some modern innovations — one of which is batch normalization. So if you want a standard Gaussian for initialization, why not just *make* it Gaussian? Sounds fanciful? But it's practical, since the whole operation is applied to the pre-activations. Let's look at its impressive results and how to implement it.

<div class="imgcap">
  <img src="/assets/2026-08-19/batchnorm_gain=1.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/batchnorm_gain=5over3.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/batchnorm_gain=4.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">12) Saturation becomes small and stable.</div>
</div>

> For backprop, the benefits are even greater. I won't show the full picture here due to length, but it's worth trying out in the [notebook](https://github.com/telingc/telingc.github.io) in the repo.

```python
# ...
g            = torch.Generator().manual_seed(2147483647)
kaiming_init = ((5 / 3) / ((block_size * n_embd) ** 0.5))
C            = torch.randn((vocab_size, n_embd),                  generator=g)
W1           = torch.randn((block_size * n_embd, n_hidd),        generator=g) * kaiming_init
b1           = torch.randn(n_hidd,                               generator=g) * 0.01
W2           = torch.randn((n_hidd, vocab_size),                 generator=g) * 0.01
b2           = torch.randn(vocab_size,                           generator=g) * 0

bngain       = torch.ones((1, n_hidd))
bnbias       = torch.zeros((1, n_hidd))
# ...
for i in range(200000):
# ...
    bnmeani  = hpreact.mean(0, keepdim=True)
    bnstdi   = hpreact.std(0, keepdim=True)
    hpreact  = bngain * (hpreact - bnmeani) / bnstdi + bnbias
# ...
```

Statistically, scaling up by a factor of $n$ enlarges the standard deviation by $n$. So if you need to center the distribution at zero, just subtract the mean of the pre-activations; if you want unit variance, just divide by the standard deviation. And most importantly, all of this is built into the neural net — meaning we can ensure a roughly standard-normal pre-activation every iteration, firing the non-linearity at full capacity, and even giving the net extra parameter knobs to tune for a better fit. Let's take a close look at these formulas, because they're simple and useful. For my tiny single-layer net, batchnorm doesn't help much, but for multi-layer nets it makes training more forgiving — WaveNet (part 5) is a case in point.

1. $$\frac{1}{m}\sum_{i=1}^{m}x_i = \mu_{\mathcal{B}} \quad (\text{the mean of the pre-activations})$$
2. $$\frac{1}{m-1}\sum_{i=1}^{m}(x_i-\mu_{\mathcal{B}})^2 = \sigma_{\mathcal{B}}^2 \quad (\text{the variance of the pre-activations})$$
3. $$\frac{x_i-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}} = \hat{x_i} \quad (\text{normalization})$$
4. $$\gamma\hat{x_i}+\beta = y_i \equiv \mathrm{BN}_{\gamma,\beta}(x_i) \quad (\text{scale and shift})$$

Implementing BatchNorm is quite straightforward with a tutorial, but there's a subtle problem lurking here. Focus on formula 3: we're computing the variance and mean from a batch, which means once we implement BatchNorm, the input is inherently a batch. So how do we feed in a single example and get a sensible result?

One intuitive idea is to simply calibrate these two values and fix them after training.

```python
# calibrate batch norm at the end of training

with torch.no_grad():
    # pass the training set through
    emb     = C[Xtr]
    embcat  = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 # + b1
    # measure the mean/std over the entire training set
    bnmean  = hpreact.mean(0, keepdim=True)
    bnstd   = hpreact.std(0, keepdim=True)
```

But that isn't what's suggested in the video. Instead, we compute a running mean and standard deviation during training. Here's the implementation:

```python
# inside the forward pass

bnmeani = hpreact.mean(0, keepdim=True)
bnstdi  = hpreact.std(0, keepdim=True)
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi

h       = torch.tanh(hpreact)
logits  = h @ W2 + b2
loss    = F.cross_entropy(logits, Yb)

# and the backward pass.
```

The video and the code leave an unstated math problem: the magical numbers 0.999 and 0.001 above, and why this gives a reasonable estimate of the mean and standard deviation. The 0.001 is called *momentum*, which we'll see again after we refactor our net into objects.

And here's the loss comparison between running statistics and fixed mean/std.

<table>
 <tr>
   <th>set \ method</th>
   <th>loss using running mean & std</th>
   <th>loss using fixed mean & std</th>
 </tr>
 <tr>
   <td>training set</td>
   <td>2.078</td>
   <td>2.079</td>
 </tr>
 <tr>
   <td>validation set</td>
   <td>2.113</td>
   <td>2.113</td>
 </tr>
 <tr>
   <td>test set</td>
   <td>2.113</td>
   <td>2.113</td>
 </tr>
 <caption>loss comparison</caption>
</table>

The losses are basically the same. Skipping the explicit calibration stage is for efficiency: when working on a larger, deeper net like GPT-3 — 175 billion parameters across 96 layers — just imagine adding a calibration pass. An explicit stage could be both time- and financially consuming.

And that's my first blog. Hope you like it.

---

### Bonus topic: pool balls

Here's something I found interesting. You don't have to read it — it's meaningless, but I spent quite a while on it and don't want it to go to waste.

Intuitively, it reminds me of elastic collision. Think of softmax: when the normalization function gets an extra parameter, *Temperature*, you can view it through the lens of the partition function.

<div class="imgcap">
  <img src="/assets/2026-08-19/momentum.jpg" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Momentum of a pool cue ball is transferred to the racked balls after collision. Found on Wikipedia.</div>
</div>

From the formula below, we can construct a physical model of two pool-ball-like particles under classical dynamics. It's not hard to show that for the collision to satisfy the equivalent equation 2), we need $\frac{m_2}{m_1} = 1999$ and $v_1 = 1\,\mathrm{m/s}$ initially.

<div class="imgcap">
  <img src="/assets/2026-08-19/elastic_collision.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">The model I imagine.</div>
</div>

$$1)\quad bnmean_{running} = 0.999 \times bnmean_{running} + 0.001 \times bnmean_{i}$$
$$2)\quad \Rightarrow v_2 = (1-0.001) v_1 + 0.001 w_1$$

So we can imagine many pool balls of mass $m_1$, each with a different velocity $w_i$, colliding head-on with $m_2$. Because $m_2$ is much heavier than $m_1$, $m_2$ will never overtake the lighter balls after the collision.

By induction, it's easy to verify that the velocity of $m_2$ after the $n$-th collision is

$$v_n = (1-momentum)^{n-1} + momentum\sum_{i=1}^{n-1}(1-momentum)^{n-1-i}w_i, \quad \text{for } n\geqslant 2$$

Since

$$1)\quad v_1 = 1 \implies v_2 = (1-momentum)^1 + (momentum)w_1$$
$$2)\quad v_{n+1} = (1-momentum)v_n + momentum \cdot w_n$$
$$\qquad\qquad = (1-momentum)^n + momentum\sum_{i=1}^{n-1}(1-momentum)^{n-i}w_i + momentum \cdot w_n$$
$$\qquad\qquad = (1-momentum)^n + momentum\sum_{i=1}^{n}(1-momentum)^{n-i}w_i$$

You can take the pool-ball analogy as an interesting coincidence, but intuitively there's a real explanation: many small balls colliding with a "mother ball" $m_2$ whose mass is much larger ($m_1 = \frac{1}{1999}m_2$). After enough collisions, the mother ball settles at roughly the mean velocity of the small balls.

So, after this bonus detour, back to BatchNorm — we can now express the mean and std of the $n$-th training step ($n \geqslant 2$):

$$bnmean_{running,n} = (1-momentum)^{n-1} + momentum\sum_{i=1}^{n-1}(1-momentum)^{n-1-i}bnmean_i$$

*momentum* is a relatively small value. At the scale of 200,000 training iterations, $(1-momentum)^{n-1}$ decays exponentially to zero. As for the other term, $momentum\sum_{i=1}^{n-1}(1-momentum)^{n-1-i}bnmean_i$ — the larger the index of a $bnmean_i$, the less weight it carries in the running average.

To choose the value of the magical *momentum*, you can use a baseline tied to the data scale — for example, $\frac{1}{1000}$ for a dataset on the order of $10^5$.
