---
layout: post
title: "Visualizing a simple neural network under the hood, from a naive view."
excerpt: "Visualizing activation, gradients & batch normalization, a brief demonstration for neural nets built in an ancient way, from a NOOB perspective. A short hike through nn-zero-to-hero."
date: 2026-08-19
comments: true
mathjax: true
---

[Makemore](https://github.com/karpathy/makemore) of [Andrej](https://karpathy.ai/) confused me quite a lot while learning part 3. If topics like BatchNorm, KaimingInit confuse you as they confused me. I hope you find this helpful.
Aimming to visualize activation, gradients & batch normalization, to briefly demonstrate neural nets built in an ancient way from a NOOB perspective. Enjoying a short hike through [nn-zero-to-hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).
Let's jump right in.

### WAAAIT, It's just a table of probabilities!
Training the bigram neural net which takes two former characters, embedded, went through the hidden layer and then softmax calculated and got the probabilities prediction of the next character could be seemingly futile. 'Cause normalized logits looks exactly same with statistics in the graph below.

That's literally my naive thoughts after drawing this graph:
> *"How about look up this table manually? Like the musing of a HLM (Human Language Model🧠🤖)"*
<div class="imgcap">
  <img src="/assets/2026-08-19/makemore_contrast.png" width="80%" style="display:block; margin:auto;">
  <div class="thecap">1) Bigram neural net visualization</div>
</div>

Looking at graph above(darker blocks means higher probability.), it's kinda natural for me to come up with the idea like, how about create this table with statistics and simply use random to look it up? What the net outputs finally is the probabilities which could be obtained use statistics.

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

But the truth is often not the case 'cause if you want to scale up and get higher efficiency and performance that predicts well even with a smaller training set, the context offered by trigram is often the better choice. Looking at the table above, eightgram WaveNet is much more *"name-like"* to me. At this point, the probabilities model looks like this:

<div class="imgcap">
  <img src="/assets/2026-08-19/cube.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">2) Trigram neural net visualization</div>
</div>

The 3-dimension heatmap on the left, representing the probabilities of under the condition of different character combinations, the next character is *a(the char)*. For example, the tiny green dot, stands for possibilities of `emm-->a` which is 28%.

$$P(\mathrm{next\,character\,is\,}a|\mathrm{context\,is\,}emm)\approx0.28$$

And there's 27 cubes like this for the simplest net with out batchnorm. even it's a silly idea, I still think it's beautiful.🥹
(Drawing this is just fun and cool, just ignore me.😄)

### High initial loss caused by improper initialization

<div class="imgcap">
  <img src="/assets/2026-08-19/lossi.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">3) loss.log10() decreasing with training</div>
</div>

As you can see, the initial loss is $\geqslant10^1.4\approx25.12$. But we have no reason to believe that any character is having more chance than others, so the loss of uniform distribution is expected, and it can be calculated as:

$$-\ln\frac{1}{27} = 3\ln3\approx3.29583687$$

I verified it in IDLE, this is how it goes:

```python
>>> import torch
>>> uni_dis = torch.ones(1, 27) * 1 / 27
>>> loss = -(uni_dis.log()).mean()
>>> loss.item()
3.295837163925171
```

> Playing this toy model, when Andrej saw this insanely high loss but he have to keep this hilarious moment 'til the next class, must be challenging.😂

What's happening is `torch.randn` generates numbers from standard normal distribution $out_i\sim\mathcal{N}(0,1)$. So after the hidden layer, tanh and softmax, the probabilities can hardly be uniformed, and sadly it's getting pretty "paranoid".

By chance, I could have a crazy low loss 'cause my net is correctly confident on a random choice, but most likely, I'll run into a big big wrong. You'll find in the graph below distribution values of C obeys gaussian roughly, but when it comes to the mean of predictions in this batch, and the chance we have to get a *t* out of five trials is "promising". The yellow line below is the uniform distribution. Comparing to the distribution we have, there's a lot to optimize.

<div class="imgcap">
  <img src="/assets/2026-08-19/gaussian.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">4) Improper initialization.</div>
</div>

How to make the `probs` more uniformed? Well, the simplest idea is find a way to decrease the variance of the data, then the object I should operate is logits of cause. Hidden layer couldn't be touched, if so, scale down `W2` and initial `b2` to zero sounds reasonable, since it's matrix multiply, scale down a bit could bring a well decrease to the loss.

```python
# ...
g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),           generator = g)
W1 = torch.randn((block_size * n_embd, n_hidd), generator = g) # * 0.1
b1 = torch.randn(n_hidd,                        generator = g) # * 0.01
W2 = torch.randn((n_hidd, vocab_size),          generator = g) * 0.01
b2 = torch.randn(vocab_size,                    generator = g) * 0
# ...
```

And the result is that the initial loss was decreased from 27.9 to 3.3, slightly higher than $3\ln3$, and the final performance was raised by a decent number(final loss: 2.15->2.13 on val set). Of great significance, efficiency, now we skipped the duration of sliding down the steep *hocky stick bar*🏒.

<div class="imgcap">
  <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px;">
    <img src="/assets/2026-08-19/lossi.png" width="45%" style="display:block; margin:auto;">
    <img src="/assets/2026-08-19/lossi-new.png" width="45%" style="display:block; margin:auto;">
  </div>
  <div class="thecap">5) loss.log10() decreasing with training</div>
</div>

### Do neural networks dream of dead neurons?
So, as Andrej says, there's a problem hiding in the simple net currently. To identify this, one line of code is needed in order to record our grads:

```python
# ---minibatch construct--->>>
ix      = torch.randint(0, Xtr.shape[0], (batch_size, ), generator = g)
Xb, Yb  = Xtr[ix], Ytr[ix]
# ---forward pass       --->>>
emb     = C[Xb]
embcat  = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1 + b1
h       = torch.tanh(hpreact)
logits  = h @ W2 + b2
loss    = F.cross_entropy(logits, Yb)
# ---backward pass      --->>>
for p in parameters:
    p.grad = None

t.retain_grad() for t in [hpreact] # <<<--- THIS IS IT!

loss.backward()
# ---update             --->>>
lr = 0.1
p.data += -lr * p.grad
```

If I just draw the histogram of this out and see the grads of pre-activation, I believe this is even more *sweating* than watching the video. Just staring at this:

1. $$\frac{\partial}{\partial x} \tanh{x}\bigg|_{x=pre-activation_{ij}}=0$$
  
2. `p.data += -lr * p.grad` 

Meaning that `pre_activation_ij.grad` is 0, and for the neuron pre_activation_ij, it will never be updated! These bad bad neurons are just like me in the university classes, sitting there and learning NOTHING. Neurons like this, we have totally 5000 more hiding behind tanh layer. It's being very kind to say that our hidden layer is a University.

<div class="imgcap">
  <img src="/assets/2026-08-19/dead_neurons.png" width="70%" style="display:block; margin:auto;">
  <div class="thecap">6) Dead neurons.</div>
</div>

*"Backward passing"* enough for now, so let's bring this question forward, and figure out where does this come from. Actually, people find this problem from activation h, there's tons of 1.00s in it. Area around $\pm1$ holds the major distribution density of values in h, if $h_{ij}=1$, then corresponding value in pre-activation.grad would be 0. BUT! you might argue, it couldn't be perfectly 1 'cause the index could never reach infinity, so why is that? Yes, and the float is not accurate enough to make sure huge numbers' tanh is not 1. The left graph below, red curve stands for the gradient while blue area is distribution. When values distributes outside the orange lines, their gradients are tend to be zero, referencing $\frac{\partial}{\partial x}\tanh{x}$.

<div class="imgcap">
  <img src="/assets/2026-08-19/density.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">7) Values are in the plateau of tanh with minor gradients.</div>
</div>

Well, I hope it's clear enough. Intuitively, totally white bars are dead neurons. In this case, the net is just inactive. If you find the picture with its face paled, you should be, too.

<div class="imgcap">
  <img src="/assets/2026-08-19/whites.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">8) Whiter the pic is, more inefficient the net is.</div>
</div>

To solve this issue, imagine squeezing the pre-activation distribution and make it thinner, thus values wont across the orange line. So scale down W1 and b1 is a good choice.

```python
# ...
g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),           generator = g)
W1 = torch.randn((block_size * n_embd, n_hidd), generator = g) * 0.1
b1 = torch.randn(n_hidd,                        generator = g) * 0.01
W2 = torch.randn((n_hidd, vocab_size),          generator = g) * 0.01
b2 = torch.randn(vocab_size,                    generator = g) * 0
# ...
```
Now it's much better. The final loss was decreased from originally 2.15 to 2.13 before and now it's 2.10. And if the distribution is too concentrated, then the $tanh$ is doing nothing.

<div class="imgcap">
  <img src="/assets/2026-08-19/correct_init.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/goodgood_neurons.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">9) Now they're motivated neurons.</div>
</div>

### The myth, the legend, Kaiming Initialization!

```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200) * magical_number
y = x @ w
print(x.mean(), y.mean())
print(x.std(),  y.std())
```
From the code above, what can we expect from mean and standard deviation?
1. magical number = 1, standard deviation grows larger.
2. magical number > 1, standard deviation grows even larger.
3. 0 < magical number < 1, standard deviation decreases.

The whole point is that finding a way to prevent the normal distribution shrinking to 0 or expand to infinity, and keep the standard deviation still be 1. [Kaiming Initialization](https://arxiv.org/abs/1502.01852) suggests the magical number should be $\sqrt{\frac{2}{fan_{in}}}$ for activation, and $\sqrt{\frac{gain^2}{fan_{in}}}$ for backward pass, gain depends on the non-linearity. In the case of $\tanh{x}$, $gain = \frac{5}{3}$. They're different because the ways they contract are quite different, so there must be different gains comes out to scale up and go against the non-linearity.

<div class="imgcap">
  <img src="/assets/2026-08-19/kaiming.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/kaiming_neurons.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">10) Now they're stable Kaiming neurons, activation was initialized.</div>
</div>

Configuring $gain$ is nontrivial, choosing $gain=\frac{5}{3}$ for initialization is considering, comparing to $gain=1$, tanh is a squashing function, making the graph tend to gravitated to 0, if $gain$ is lower, it would be harder to fight against the tendency.

And for $gain=4$, the activation tend to be way to saturated.

<div class="imgcap">
  <img src="/assets/2026-08-19/gain=1.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/gain=5over3.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/gain=4.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">11) Different gains.</div>
</div>

Once we done with BatchNorm, the activation will not be so sensitive.

> The math behind it, I'd like to leave it just for now.

### Confusing BatchNorm
Precisely configured and fragile initializations, are less important today due to some modern innovations, one of them is batch normalization. So if you want a normal gaussian for initialization, why not just make it gaussian? Sounds imaginative? But this is practical since this full operation is just about pre-activation initializations. Let's take a look at it's awesome outcomes and see how to implement it.

<div class="imgcap">
  <img src="/assets/2026-08-19/batchnorm_gain=1.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/batchnorm_gain=5over3.png" width="100%" style="display:block; margin:auto;">
  <img src="/assets/2026-08-19/batchnorm_gain=4.png" width="100%" style="display:block; margin:auto;">
  <div class="thecap">12) Saturation become small and stable.</div>
</div>

> for backprops, the benefits are even more, I'm not going to show it fully here due to the length, but it's a nice choice to try it in the [notebook](https://github.com/telingc/telingc.github.io) in the repo.

```python
# ...
g            = torch.Generator().manual_seed(2147483647)
kaiming_init = ((5 / 3) / ((block_size * n_embd) ** 0.5))
C            = torch.randn((vocab_size, n_embd),           generator = g)
W1           = torch.randn((block_size * n_embd, n_hidd), generator = g) * kaiming_init
b1           = torch.randn(n_hidd,                        generator = g) * 0.01
W2           = torch.randn((n_hidd, vocab_size),          generator = g) * 0.01
b2           = torch.randn(vocab_size,                    generator = g) * 0

bngain       = torch.ones((1, n_hidd))
bnbias       = torch.zeros((1, n_hidd))
# ...
for i in range(200000):
# ...
    bnmeani  = hpreact.mean(0, keepdim=True)
    bnstdi   = hpreact.std(0, keepdim=True)
    hpreact  = bngain * (hpreact - bnmeani) / bnstdi - bnbias
# ...
```

For statistcs, scale up n times makes standard deviation enlarged n times, thus if you need to make the distribution aligned with zero, just minus the mean of the pre-activation, if you want the variance to be one, just devide by standard deviation. And most importantly, these are all took into our neural net, which means we can ensure to gain a roughly standard normal-like pre-activation every round, firing the non-linearity's all might, and even leave the net tuning more parameter knobs to fit the model better. Let's take a close look at these formulae, just because they're easy and useful. For my single layer tiny net, it's not gaining much from batchnorm, but when it comes to multi-layer nets, that would make the net a bit forgiving, WaveNet(part 5) is the case.

1. $$\frac{1}{m}\sum_{i=1}^{m}x_i=\mu_{\mathcal{B}}\quad(\mathrm{the\,mean\,of\,the\,pre-activation.})$$
2. $$\frac{1}{m-1}\sum_{i=1}^{m}(x_i-\mu_{\mathcal{B}})^2=\sigma_{\mathcal{B}}\quad(\mathrm{the\,variance\,of\,the\,pre-activation.})$$
3. $$\frac{x_i-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}}=\hat{x_i}\quad(\mathrm{the\,normalization})$$
4. $$\gamma\hat{x_i}+\beta=y_i\equiv\mathrm{BN}_{\gamma,\beta}(x_i)\quad(\mathrm{scale\,and\,shift})$$

Implementing BatchNorm is quite relaxing with tutorial, but there's one problem hiding behind this vibe, concentrating on formula 3, we're now calculating with variance and mean that means once we implement BatchNorm, the input now become a batch. So how to feed in a single example and get a sensible result out?

One intuitive idea is simply calibrate these two value and make them fixed after training. 

```python
# calibrate the batch norm at the end of training

with torch.no_grad():
  # pass the training set through
    emb     = C[Xtr]
    embcat  = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 # + b1
    # measure the mean/std over the entire training set
    bnmean  = hpreact.mean(0, keepdim=True)
    bnstd   = hpreact.std(0, keepdim=True)
```

But that isn't suggested in the video, instead we're calculating mean and standard deviation running along the training. And here is implementation:

```python
# before it is forward pass.

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

The video and the code left us a unstated math problem, the magical number $0.999$ and $0.001$ above and why this is able to calculate the mean and standard deviation roughly. the $0.001$ in it was called *momentum*, which we'll see after we summarize our net into objects.



And here is the loss comparison between running calculation and fixed mean&std.

<table>
 <tr>
   <th>set\method</th>
   <th>loss using running mean&std</th>
   <th>loss using fixed mean&std</th>
 </tr>
 <tr>
   <td>training set</td>
   <td>2.078</td>
   <td>2.079</td>
 </tr>
 <tr>
   <td>validate set</td>
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

The losses of the two are basically the same, skipping the explicit stage of calibration is for efficiency, when working on a larger and deeper net like GPT-3 who have 175 billion parameters in 96 layers, just imagine adding a calibration stage for the model, a explicit stage may be both time and financial consuming.

And that's my first blog, hope you like it.

---
### Bonus topic: pool balls
And here is what I found interesting, but there's no need to read. It's meaningless but I still spend quite a while and don't want to waste it.

Intuitively, it reminds me to consider from the view of elastic collision, like when softmax, the normalization function was added another parameter *Temprature*, then you can view from the point of partition function.

<div class="imgcap">
  <img src="/assets/2026-08-19/momentum.jpg" width="50%" style="display:block; margin:auto;">
  <div class="thecap">Momentum of a pool cue ball is transferred to the racked balls after collision. Found on wikipedia.</div>
</div>

From the formula below, we could construct a physical model consists of two pool ball-like particles, under the view of classic dynamics. It's not very tricky to infer that to make the collision satisfy the equivalent equation 2), $\frac{m_2}{m_1} = 1999$ and $v_1 = 1\mathrm{m/s}$ initially.

<div class="imgcap">
  <img src="/assets/2026-08-19/elastic_collision.png" width="50%" style="display:block; margin:auto;">
  <div class="thecap">The model I imagine.</div>
</div>

$$1)\quad bnmean_{running} = 0.999 \times bnmean_{running} + 0.001 \times bnmean_{i}$$
$$2)\quad \Rightarrow v_2 = (1-0.001) v_1 + 0.001 w_1$$

Thus we could imagine that there's numerous pool balls like $m_1$ having different velocity $w_i$ but same mass $m_1$, colliding with $m_2$ from the front, 'cause the mass of $m_2$ is much larger than $m_1$, thus $m_2$ will never catch the ball weights $m_1$ after the collision.
Inductionally, it's easy to verify the velocity of $m_2$ after nth collision is

$$v_n = (1-momentum)^{n-1}+momentum\sum_{i=1}^{n-1}(1-momentum)^{n-1-i}w_i,\, for\,n\geqslant 2$$

Due to

$$1)\quad v_1 = 1\,\, thus\,\, v_2 = (1-momentum)^1+(momentum)w_1$$
$$2)\quad v_{n+1} = (1-momentum)v_n+momentum \cdot w_n$$
$$\qquad\qquad=(1-momentum)^n + momentum\sum_{i=1}^{n-1}(1-momentum)^{n-i}w_i + momentum \cdot w_n$$
$$\qquad\qquad=(1-momentum)^n + momentum\sum_{i=1}^{n}(1-momentum)^{n-i}w_i$$

You can take the pool balls example as a interesting coinstance, but intuitively there's an explanation that numerous balls colliding with the mother ball $m_2$ whose mass is bigger than small tiny balls weighting $m_1=\frac{1}{1999}m_2$. After a while, the mother ball will have the mean velocity of the small balls.

So, after some bonus talks, back to our BatchNorm, we could now calculate the mean and std of the nth training ($n\geqslant2$).

$$bnmean_{running,n} = (1-momentum)^{n-1}+momentum\sum_{i=1}^{n-1}(1-momentum)^{n-1-i}bnmean_i$$

$momentum$ is relatively a small value, under the scale of 200000 times of training, $(1-momentum)^{n-1}$ decrease exponentially to zero. The another factor $momentum\sum_{i=1}^{n-1}(1-momentum)^{n-1-i}bnmean_i$, for $bnmean$s, the bigger the index is, its significance to $bnmean_{running}$ become less and less.

To decide the value of the magical number $momentum$, we could have a baseline to compress the data, like $\frac{1}{1000}$ for dataset scaling $10^5$ for example.

---
