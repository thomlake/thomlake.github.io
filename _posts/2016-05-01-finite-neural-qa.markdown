---
layout: post
title:  "Neural Question Answering with Finite Memory"
date:   2016-05-01
tags : [memnet, bAbI, neural-networks, nlp, lstm]
summary: Neural QA systems predominantly rely on a memory space whose size scales linearly with the number of facts. This post describes preliminary experiments with an architecture capable of answering simple questions by maintaining a finite set of memory locations as it iteratively reads a sequence of facts.
---

## Introduction
Recent work has shown that despite historical opinion to the contrary[^connectionism], neural networks can learn to perform several tasks requiring symbolic manipulation. This includes answering questions by reasoning over large knowledge bases[^large-scale-qa], combinatorial optimization[^pointer-nets], sorting and copying[^ntm], and manipulating data structures such as stacks and queues[^transduce]<sup>,</sup>[^stack-rnn]. 

Unlike most recent deep learning successes which primarily rely on a combination of increased depth, better initialization and optimization, larger datasets, and increased computing power due to the use of GPUs[^deep-learning-nature], the above mentioned capabilities are largely driven by architectural modifications and the introduction of novel neural network components. 

Typically these new components are derived by defining differentiable smooth versions of symbolic operations. For example, replacing an all-or-nothing read operation with a state dependent convex combination[^convex-combination] yields the attention component popular in NLP[^attention] and Vision[^draw] applications. Since these new components are differentiable they can be combined with traditional dense, convolution, and recurrent neural network layers to obtain a unified system whose parameters can be trained using the gradient based optimization algorithms which have been used to train neural networks for decades. 

Inspired by this line of research, in particular Memory Networks[^memnn] (MemNNs) and Neural Turing Machines[^ntm] (NTMs), this work describes and analyzes the behavior of a neural network architecture consisting of a finite set of memory locations (registers) coupled with a mechanism to mediate reading and writing from them.
<!-- and a desire to better understand the inductive bias conferred by these architectures,  -->

<!-- ## Memory Structures for Neural Networks
The canonical 

$$ 
    \mathbf{h}_t = \sigma\left(
        \mathbf{W}\mathbf{x}_t + \mathbf{U}\mathbf{h}_{t-1} + \mathbf{b}
    \right)
$$

The memory capacity (hidden state size) and the number of parameters in a recurrent neural network (RNN) are highly coupled. A standard RNN[^srnn] with \\(n\\) hidden units and \\(m\\) inputs has \\(m(m + n)\\) parameters, a RNN with Long Short-Term Memory[^lstm] (LSTM) units has \\(4m(m + n)\\) parameters, and a RNN with Gated Recurrent units[^gru] has \\(3m(m+n)\\) parameters[^fn-no-peep]. 

This coupling is highly undesirable since it arbitrarily ties model complexity, as measured by number of learnable parameters, with the ability to store large amounts of information. Broadly speaking, to store lots of information in a RNN you need lots of parameters and to fit lots of parameters you need lots of data. 

Recent works have decoupled model complexity and storage capacity by introducing external memory structures which can be read an written to. Examples include Memory Networks[^memnn]'[^end2end] (MemNNs), Neural Turing Machines[^ntm] (NTMs), and Stack-Augmented RNNs[^stack-rnn].  -->


## Register Networks
Register Networks (RegNets) are a neural network architecture featuring a fixed number $$ K $$ of finite dimensional registers coupled with a stateful control mechanism. During the input phase at time $$ t $$ the network observes an input $$ x_t \in \mathcal{X} $$, controller state $$ s_t \in \mathcal{S} $$, and registers $$ \mathbf{m}^1_t, \ldots, \mathbf{m}^K_t $$, $$ \mathbf{m}^i_t \in \mathbb{R}^d $$, and ouputs a new controller state $$ \mathcal{s}_{t+1} $$ and updated registers $$ \mathbf{m}^1_{t+1}, \ldots, \mathbf{m}^K_{t+1} $$. During the output phase the network observes a query $$ q \in \mathcal{Q} $$ and the current state of the registers and produces an output $$ \mathcal{y} \in \mathcal{Y} $$. Thus, at a high level a RegNet can be defined in terms of an input and output function

$$
    \begin{align*}
        & f_i \colon
            & \mathcal{X} \times \mathcal{S} \times \mathbb{R}^{d \times K}
            & \longrightarrow \mathcal{S} \times \mathbb{R}^{d \times K}
            & \text{input} \\
        & f_o \colon 
            & \mathcal{Q} \times \mathbb{R}^{d \times K} 
            & \longrightarrow \mathcal{Y}
            & \text{output} \\
    \end{align*}
$$

### Architecture details
Let $$ x_t, q_t, s_t, \mathbf{m}^1_t, \ldots, \mathbf{m}^K_t $$ denote the input, query, controller state, and memory contents at time \\(t\\). A LSTM[^lstm] (without peephole connections) was used as the controller and therefore $$ s_t = (\mathbf{h}_t, \mathbf{c}_t) $$ denotes the hidden and cell states respectively. A linear input embedding with parameters $$ \mathbf{A} \in \mathbb{R}^{m \times n} $$ is denoted by the function $$ \phi_\mathbf{A} $$. For example if $$ x_t $$ is a bag-of-words then $$ \phi_\mathbf{A}(x_t) = \sum_{w \in x_t} (\mathbf{A})_{:,w} $$. Figure 1 depicts a single RegNet input step.

<figure>
<center>
<img src="{{ site.baseurl }}/assets/finite-neural-qa/regnet-diagram.png" height="500px" />
<figcaption><b>Figure 1:</b> Graphical representation of RegNet input step.</figcaption>
</center>
</figure>

**State Controller:**
Given the current input, previous controller state, and previous register contents the controller updates the state and registers according to the following equations:

$$
    \begin{align*}
        s_t 
            &= (\mathbf{h}_t, \mathbf{c}_t) = \texttt{lstm}(\phi_\mathbf{A}(x_t), s_{t-1}) \\
        \mathbf{e} 
            &= σ(\mathbf{W}_e \phi_\mathbf{B}(x_t) + \mathbf{b}_e) \\
        \mathbf{a} 
            &= σ(\mathbf{W}_a \phi_\mathbf{B}(x_t) + \mathbf{b}_a) \\
        \mathbf{g} 
            &= \texttt{softmax}(\mathbf{W}_g \mathbf{h}_t + \mathbf{b}_g) \\
        \mathbf{m}^k_t 
            &= g_i \mathbf{a} + (\mathbf{1} - g_i\mathbf{e}) \odot \mathbf{m}^i_{t-1} 
            & i = 1, \ldots, K\\
    \end{align*}
$$

**Reading:**
Given a query the current register contents are summarized according to the following equations:

$$
    \begin{align*}
        \mathbf{k} 
            &= \texttt{tanh}(\mathbf{W}_k \phi_\mathbf{C}(q_t) + \mathbf{b}_k) \\
        u_i 
            &= \mathbf{k}^T \mathbf{m}^i_t 
            & i = 1, \ldots, K \\
        \mathbf{g}
            & =\texttt{softmax}(\mathbf{u}) \\
        \mathbf{r}_t 
            &= \sum_i g_i \mathbf{m}^i_t \\
    \end{align*}
$$

**Output:**
Finally, given the read vector the network output is given by:

$$
    \begin{align*}
        \mathbf{o}_t = \texttt{softmax}({\mathbf{W}_o} \mathbf{r}_t + \mathbf{b}_o)\\
    \end{align*}
$$

### Related Architectures
Like MemNNs the operation of a RegNet is broken into separate input and output phases, but unlike MemNNs which have a memory size equal to the number of inputs, RegNets have a fixed size memory necessitating the controller to facilitate updating the memory. In this sense RegNets are similar to NTMs which also have a collection of finite memory elements and a mechanism for updating them. 

However RegNets differ from NTMs in a number of ways. The most significant difference is the controller state does not depend on the contents of the external memory. This independence forces the controller to maintain some information about the contents of the registers in order to produce sensible weightings, but relieves it of the need to store all information necessary for solving the problem at hand since the output mechanism only interacts directly with the registers. In a sense the controller implements an implicit form of variable binding. 

Furthermore rather than using a combination of content and location based focusing like NTMs, write weights in a RegNet are output directly by the controller. The price paid for the simplified gating mechanism is the introduction of a mild dependence between the number of parameters (the size of the $$ \mathbf{W}_g $$ matrix and $$ \mathbf{b}_g $$ vector) and the number of registers.

Lastly it should be noted that the content based read mechanism of a RegNet is the same as that used by End-to-End MemNNs[^end2end].

## QA Experiment
Initial experiments were carried out using a modified version of task 1 from the [bAbI](https://research.facebook.com/researchers/1543934539189348) dataset[^bAbI]. The dataset was modified such that each story contains references to exactly two agents, a random number of clauses uniformly chosen from five to ten, and two questions. The first element in each story is always a clause, the last element is always a question, and the location of the other question is randomly chosen. Clauses are balanced such that the number of clauses referring to each agent is approximately equal and the subject of each question is chosen randomly. Six agents and six locations were used. The number of questions per dataset (1000) is kept the same as the original bAbI dataset resulting in 500 train and 500 test stories (2 questions per story). An example story is given in Figure 2.

The use of this modified dataset was motivated by a desire to better understand the register update strategies the network is able to learn. Setting the number of registers equal to the number of agents in each story (two) ensures a high performing controller network must bind agents to arbitrary register locations rather than using a positional encoding where each agent is always mapped to the same register, as this later strategy would likely fail when presented with a story containing agents assigned the same register.
<br /><br />

<figure>
<pre style="
    width: auto;
    background-color: #424242;
    color: #fdfdfd;
    border: none;
    border-radius: 0;
    padding: 10px;
    margin: 0;
    line-height: 25px;
    font-size: 18px;
">
1 mary journeyed to the office
2 james journeyed to the hallway
3 where is james    hallway
4 james moved to the bedroom
5 james journeyed to the hallway
6 mary went to the bathroom
7 james went back to the garden
8 mary went back to the hallway
9 where is james    garden
</pre>
<center>
<figcaption><b>Figure 2:</b> Example story from the modified bAbI dataset.</figcaption>
</center>
</figure>

### Baselines
A LSTM based baseline was used for comparison. Inputs are processed by a linear embedding and then fed to the LSTM. Queries are also processed by embedding (with a different matrix) and are then fused with the current LSTM hidden state using a Multilayer Perceptron (MLP) with a single $$ \texttt{tanh} $$ hidden layer.

$$
    \begin{align*}
        s_t 
            &= (\mathbf{h}_t, \mathbf{c}_t) = \texttt{lstm}(\phi_\mathbf{A}(x_t), s_{t-1}) \\
        \mathbf{z} 
            &= \texttt{tanh}(\mathbf{W} \phi_\mathbf{B}(q_t) + \mathbf{U}\mathbf{h}_t + \mathbf{b}) \\
        \mathbf{o}_t
            &= \texttt{softmax}(\mathbf{V} \mathbf{z} + \mathbf{c}) \\
    \end{align*}
$$

The proposed RegNet is also compared to an architecture using content based updating. In this model the state controller update equations described for the RegNet above are replaced by the following
equations:

$$
    \begin{align*}
        \mathbf{k} 
            &= \texttt{tanh}(\mathbf{W}_u \phi_\mathbf{A}(x_t) + \mathbf{b}_{u}) \\
        \mathbf{e} 
            &= σ(\mathbf{W}_e \phi_\mathbf{B}(x_t) + \mathbf{b}_e) \\
        \mathbf{a} 
            &= σ(\mathbf{W}_a \phi_\mathbf{B}(x_t) + \mathbf{b}_a) \\
        u_i 
            &= \mathbf{k}^T \mathbf{m}^i_t 
            & i = 1, \ldots, K \\
        \mathbf{g} 
            &= \texttt{softmax}(\mathbf{u}) \\
        \mathbf{m}^k_t 
            &= g_i \mathbf{a} + (\mathbf{1} - g_i\mathbf{e}) \odot \mathbf{m}^i_{t-1} 
            & i = 1, \ldots, K\\
    \end{align*}
$$

This model closely resembles a NTM with a feedforward controller (FF-NTM).

### Details
All models were fit using RMSProp[^rmsprop] with a learning rate of $$ 0.01 $$ and a decay rate of $$ 0.9 $$. After every epoch the learning rate is multiplied by $$ 0.97 $$. Adding gradient noise during training[^grad-noise] was found to be helpful while optimizing the RegNet but did not help with the LSTM or FF-NTM. All embeddings were $$ 20 $$ dimensional. The RegNet and FF-NTM both had 30 dimensional registers and a 15 dimensional LSTM controller was used for the RegNet. Results are presented for the LSTM baseline with 15, 30, and 60 hidden units. All hyperparameters were selected by random search on a small subset of the training dataset. Despite having a 75 dimensional state vector[^param-count] the RegNet has less parameters than both the 30 and 60 dimensional LSTM (Table 1).
<br /><br />

<figure>
<center>
<table class="nice-table" style="width: 60%; margin: 0;">
<tr>
<th>Model</th>
<th>Parameters</th>
<th>Test Error</th>
</tr>
<tr>
<td>LSTM (15)</td>
<td>3666</td>
<td>49%</td>
</tr>
<tr>
<td>LSTM (30)</td>
<td>7881</td>
<td>47%</td>
</tr>
<tr>
<td>LSTM (60)</td>
<td>21711</td>
<td>46%</td>
</tr>
<tr>
<td>FF-NTM</td>
<td>3996</td>
<td>11%</td>
</tr>
<tr>
<td>RegNet</td>
<td>5558</td>
<td>6%</td>
</tr>
</table>
<figcaption><b>Table 1:</b> Model comparison.</figcaption>
</center>
</figure>

## Results
Figure 3 shows training curves for the RegNet, FF-NTM, and LSTM baselines. Although all the models achieve similar training error rates (at or near 0) they exhibit drastically different behavior on the test set. In terms of generalization capabilities the LSTM baselines are significantly outperformed by the FF-NTM, which itself is outperformed by the RegNet. Furthermore the RegNet first reaches its final test error twice as fast as the FF-NTM (epoch 19 vs epoch 39).

<figure>
<center>
<img src="{{ site.baseurl }}/assets/finite-neural-qa/training-curves.png" />
<figcaption><b>Figure 3:</b> Training curves for RegNet and LSTM. Dashed and solid lines depict errors on the train and test set respectively.</figcaption>
</center>
</figure>

## Discussion
The results above suggest that the structure of the FF-NTM and RegNet serves as a powerful inductive bias which promotes learning solutions which generalize well, unlike the more homogeneous LSTM baselines which clearly overfit. Whether the LSTM baselines could be improved through the use of regularization techniques such as weight norm penalties or dropout[^dropout]<sup>,</sup>[^rnn-drop] is left as future work.

<!-- Despite the lack of a stateful control mechanism the FF-NTM in principle has all necessary information to solve the problem at hand. An obvious solution would be to have the key vectors for both update and reading focus exclusively on the agents and the add and erase vectors act in tandem to simply overwrite the existing register contents with the current agent and location information.  -->

The difference in performance between the RegNet and FF-NTM is more difficult to explain. Lacking a strong single hypothesis, other than the vague notion that the FF-NTM was unable to learn use the external memory as effectively as the RegNet for some reason, two hypothesis are put forward. 

### Hypothesis 1: Diffuse Register Weights
Intuitively one would expect individual read and write weights to be highly focused on a single register location. On the other hand the average weighting for each agent should be diffuse since the register each agent is assigned should depend on the order in which they appear in the story, not the identity of the agent. 

These intuitive ideas can be formalized using the information theoretic concept of entropy.[^shannon-entropy] To this end let $$ \mathcal{W} $$ be the set of write (or read) weightings. Likewise let $$ \mathcal{W}_i \subseteq \mathcal{W} $$, $$ i = 1, \ldots, N_a $$ be the set of weightings when the subject of the clause (or query) is agent $$ i $$. Define the conditional expected weight for agent $$ i $$ as

$$
\begin{align*}
    \bar{\mathbf{w}}_i &= \frac{1}{|\mathcal{W}_i|} \sum_{\mathbf{w} \in \mathcal{W}_i} \mathbf{w}
\end{align*}
$$

Then the average entropy of the conditional expected agent weights is given by

$$ 
\begin{align*}
    a &= \frac{1}{N_a} \sum^{N_a}_{i=1} \text{H}(\bar{\mathbf{w}}_i)
\end{align*}
$$

and the average weight entropy is

$$
\begin{align*}
    h &= \frac{1}{|\mathcal{W}|} \sum_{\mathbf{w} \in \mathcal{W}} \text{H}(\mathbf{w}) \\
\end{align*}
$$

where $$ \text{H} $$ is Shannon's entropy function

$$
\begin{align*}
    \text{H}(\mathbf{w}) &= -\sum_{i} w_i \log w_i
\end{align*} 
$$

Connecting these quantities back to the intuitive notions discussed above one would like would $$ a $$ to be large, $$ h $$ to be small, and therefore the ratio $$ \frac{a}{h} $$ to be large.

<figure>
<center>
<img src="{{ site.baseurl }}/assets/finite-neural-qa/regnet-agent-weights.png" />
<figcaption><b>Figure 4:</b> Expected agent write weights for RegNet.</figcaption>
</center>
</figure>
<br />

<figure>
<center>
<img src="{{ site.baseurl }}/assets/finite-neural-qa/ff-ntm-agent-weights.png" />
<figcaption><b>Figure 5:</b> Expected agent write weights for FF-NTM.</figcaption>
</center>
</figure>
<br />

Figures 4 and 5 depict the conditional expected write weight for each agent. Qualitatively these figures indicate that there is less certainty regarding which location will be focused on given a particular agent in the FF-NTM than the RegNet, which agrees with the quantitative values in Table 2. 

However, as can also be seen in Table 2 this is true in general of the write weightings produced by the FF-NTM model, which have higher average entropy than the RegNet. From this it can be concluded that although the RegNet is less location agnostic than the FF-NTM, the RegNet does a superior job of focusing on a particular location. Interestingly this same relationship seems to carry over to the read weights as well suggesting this is not merely an artifact of the content based similarity weighting.
<br /><br />

<figure>
<center>
<table class="nice-table" style="width: 60%; margin: 0;">
<tr>
<th></th>
<th>FF-NTM (read)</th>
<th>RegNet (read)</th>
<th>FF-NTM (write)</th>
<th>RegNet (write)</th>
</tr>
<tr>
<td style="text-align:left; padding-left:15px; border-right: solid black 2px">$$ a $$</td>
<td>0.658</td>
<td style="border-right: solid black 2px">0.478</td>
<td>0.663</td>
<td>0.498</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px; border-right: solid black 2px">$$ h $$</td>
<td>0.138</td>
<td style="border-right: solid black 2px">0.114</td>
<td>0.193</td>
<td>0.134</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px; border-right: solid black 2px">$$ \frac{a}{h} $$</td>
<td>4.767</td>
<td style="border-right: solid black 2px">4.191</td>
<td>3.431</td>
<td>3.728</td>
</tr>
</table>
<figcaption><b>Table 2:</b> Read and write weight entropies.</figcaption>
</center>
</figure>

### Hypothesis 2: Error Correction
Another possible explanation for the observed performance difference is that the stateful control mechanism of the RegNet endues it with an ability to correct previous errors when writing to the registers. 

Let $$ \mathbf{w}_{ij} $$ and $$ s_{ij} $$ be the write weights and agent for the $$ j^{th} $$ clause (question) in story $$ i $$, $$ \mathcal{A}_i = \{ s_{ij} \mid j = 1, \ldots, N_i \} $$ be the set of agents appearing in any clause (question) in story $$ i $$, and define the set of write (read) positions for agent $$ a $$ in story $$ i $$ as

$$
    \mathcal{P}_i(a) = \{\arg\max(\mathbf{w}_{ij}) \mid s_{i,j} = a \}
$$

Then a write (read) error is said to occur in story $$ i $$ whenever

$$ 
    \exists a \in \mathcal{A}_i \; s.t. \; \left| \mathcal{P}_i(a) \right| > 1 
$$

Lastly the term _Story Error_ will be used to refer to the number of incorrectly answered questions in a given story. Table 2 gives the empirical probability of several read, write, and story error related events.
<br /><br />

<figure>
<center>
<table class="nice-table" style="width: 60%; margin: 0;">
<tr>
<th style="text-align:left; padding-left:15px;">Event</th>
<th>FF-NTM</th>
<th>RegNet</th>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Story Error > 0</td>
<td>0.206</td>
<td>0.084</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Story Error = 1</td>
<td>0.172</td>
<td>0.080</td>
</tr>
<td style="text-align:left; padding-left:15px;">Story Error = 2</td>
<td>0.034</td>
<td>0.004</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Read Error</td>
<td>0.056</td>
<td>0.054</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Read Error | Story Error > 0</td>
<td>0.146</td>
<td>0.095</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Read Error | Story Error = 2</td>
<td>0.059</td>
<td>0.000</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Write Error</td>
<td>0.416</td>
<td>0.522</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Write Error | Story Error > 0</td>
<td>0.864</td>
<td>0.738</td>
</tr>
<tr>
<td style="text-align:left; padding-left:15px;">Write Error | Story Error = 2</td>
<td>0.941</td>
<td>0.500</td>
</tr>
</table>
<figcaption><b>Table 2:</b> Model error breakdown.</figcaption>
</center>
</figure>
<br />

Table 2 reveals several interesting properties. First, a much smaller percentage of the RegNets story errors occur in stories having two errors compared to the FF-NTM (5% vs 17%). Second, despite making less mistakes overall the RegNet is actually more likely to make a write error than the FF-NTM. These observations provide support for the hypothesis that the stateful control mechanism allows the RegNet to recover more easily from mistakes made when writing to the registers. Figures 6 and 7 depict the write weight behavior of the RegNet and FF-NTM on a story where both commit a write error and provides a qualitative example of this behavior.

<figure>
<center>
<img src="{{ site.baseurl }}/assets/finite-neural-qa/regnet-error-story-register-weight.png" />
<figcaption><b>Figure 6:</b> Write weights for a story in which the RegNet makes a write error.</figcaption>
</center>
</figure>

<figure>
<center>
<img src="{{ site.baseurl }}/assets/finite-neural-qa/ff-ntm-error-story-register-weight.png" />
<figcaption><b>Figure 7:</b> FF-NTM write weights for the same story depicted for in Figure 6. The FF-NTM makes two errors in this story.</figcaption>
</center>
</figure>

## Conclusion
Neural QA with finite memory is a fairly unexplored area and although preliminary, the results presented above are very promising. In the future it will be interesting to explore the performance of similar architectures on the original bAbI tasks as well as real world QA data sets.

It is also worth noting that the introduction of attention mechanisms into the neural network toolbox brings with it the ability to inspect the internal behavior of the neural network in new ways. This is highly welcome in a field which has been criticised for producing uninterpretable black boxes.


## Notes and References
[^PSS]: ["Physical symbol system."](https://en.wikipedia.org/wiki/Physical_symbol_system) Wikipedia

[^convex-combination]: ["Convex_combination"](https://en.wikipedia.org/wiki/Convex_combination) Wikipedia

[^shannon-entropy]: ["Entropy (information theory)"](https://en.wikipedia.org/wiki/Entropy_(information_theory)) Wikipedia

[^connectionism]: James Garson. ["Connectionism"](http://plato.stanford.edu/archives/spr2015/entries/connectionism/), The Stanford Encyclopedia of Philosophy (Spring 2015 Edition).

[^resnet]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. ["Deep Residual Learning for Image Recognition."](http://arxiv.org/abs/1512.03385) arXiv preprint arXiv:1512.03385 (2015).

[^deep-learning-nature]: Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. ["Deep learning."](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) Nature 521.7553 (2015): 436-444.

[^distributed-compositional]: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeff Dean. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.

[^srnn]: Jeffrey Elman. ["Finding structure in time."](http://onlinelibrary.wiley.com/store/10.1207/s15516709cog1402_1/asset/s15516709cog1402_1.pdf?v=1&t=inf8mh3c&s=814f6d91188800e1e6187c529245871f72f9d2ae) Cognitive science 14.2 (1990)

[^lstm]: Sepp Hochreiter and Jürgen Schmidhuber. ["Long short-term memory."](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) Neural computation 9.8 (1997): 1735-1780.

[^gru]: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. ["Empirical evaluation of gated recurrent neural networks on sequence modeling."](http://arxiv.org/abs/1412.3555) arXiv preprint arXiv:1412.3555 (2014).

[^fn-no-peep]: These calculations ignore biases, initial hidden states, and assume the more common modern LSTM variant without peep-hole connections.

[^memnn]: Jason Weston, Sumit Chopra, and Antoine Bordes. ["Memory networks."](http://arxiv.org/abs/1410.3916) arXiv preprint arXiv:1410.3916 (2014).

[^ntm]: Alex Graves, Greg Wayne, and Ivo Danihelka. ["Neural turing machines."](http://arxiv.org/abs/1410.5401) arXiv preprint arXiv:1410.5401 (2014).

[^transduce]: Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blunsom. ["Learning to transduce with unbounded memory."](https://papers.nips.cc/paper/5648-learning-to-transduce-with-unbounded-memory.pdf) Advances in Neural Information Processing Systems. (2015).

[^stack-rnn]: Armand Joulin and Tomas Mikolov. ["Inferring algorithmic patterns with stack-augmented recurrent nets."](http://arxiv.org/abs/1503.01007) Advances in Neural Information Processing Systems (2015).

[^attention]: Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. ["Neural machine translation by jointly learning to align and translate."](http://arxiv.org/abs/1409.0473) arXiv preprint arXiv:1409.0473 (2014).

[^draw]: Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, and Daan Wierstra. ["DRAW: A recurrent neural network for image generation."](http://arxiv.org/abs/1502.04623) arXiv preprint arXiv:1502.04623 (2015).

[^end2end]: Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. ["End-to-end memory networks."](http://arxiv.org/abs/1503.08895) Advances in Neural Information Processing Systems. 2015.

[^goldilocks]: Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston ["The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations."](http://arxiv.org/abs/1511.02301) arXiv preprint arXiv:1511.02301 (2015).

[^large-scale-qa]: Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston ["Large-scale simple question answering with memory networks."](http://arxiv.org/abs/1506.02075) arXiv preprint arXiv:1506.02075 (2015).

[^pointer-nets]: Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. ["Pointer networks."](http://arxiv.org/pdf/1506.03134v1.pdf) Advances in Neural Information Processing Systems. 2015.

[^fn1]: I'm going to consider less than two years recent. Given the breakneck speed at which the deep learning community is moving others may not.

[^bAbI]: [This post]({% post_url 2016-03-20-deconstructing-babi-task-1 %}) contains a detailed breakdown of bAbI task 1.

[^rmsprop]: Tijmen Tieleman and Geoffrey Hinton. [Lecture 6.5](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) COURSERA: Neural Networks for Machine Learning.

[^grad-noise]: Arvind Neelakantan, Luke Vilnis, Quoc V. Le, Ilya Sutskever, Lukasz Kaiser, Karol Kurach, and James Martens. ["Adding Gradient Noise Improves Learning for Very Deep Networks."](http://arxiv.org/abs/1511.06807) arXiv preprint arXiv:1511.06807 (2015).

[^param-count]: Two 30 dimensional registers plus 15 LSTM units.

[^dropout]: Geoffrey Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov ["Improving neural networks by preventing co-adaptation of feature detectors."](http://arxiv.org/abs/1207.0580) arXiv preprint arXiv:1207.0580 (2012).

[^rnn-drop]: Taesup Moon, Heeyoul Choi, Hoshik Lee, and Inchul Song. ["RnnDrop: A Novel Dropout for RNNs in ASR."](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf) Automatic Speech Recognition and Understanding (ASRU) (2015).

