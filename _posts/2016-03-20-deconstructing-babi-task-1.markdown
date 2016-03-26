---
layout:  post
title:   "Deconstructing bAbI Task 1"
date:    2016-03-20 
tags:    [nlp, bAbI, dataset, qa]
summary: Overview and summary statistics for bABI task 1.
---

<link rel='stylesheet' type='text/css' href='{{ site.baseurl }}/assets/deconstructing-babi-task-1/post.css' />

In 2015 [FAIR](https://research.facebook.com/ai) released the bAbI dataset. The dataset consists of 20 synthetic question answering tasks that require reasoning about agents, locations, objects, and intentions. Each instance (story) consists of a sequence of clauses and questions. A full description of the dataset, motivation, and results for several system can be found in the paper [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/abs/1502.05698).
<!-- <div style="width: 50%; text-align: justify; text-justify: inter-word;">
In 2015 <a href="https://research.facebook.com/ai">FAIR</a> released the bAbI dataset. The dataset consists of 20 synthetic question answering tasks that require reasoning about agents, locations, objects, and intentions. Each instance consists of a sequence of statements and questions. A full description of the dataset, motivation, and results for several system can be found in the paper <a href="http://arxiv.org/abs/1502.05698">Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks</a>.
</div> -->

## Example Instance

<div class="fill-box" style="padding-right: 0;">
<pre class="plain">
 1 Mary moved to the bathroom.
 2 John went to the hallway.
 3 Where is Mary?    bathroom    1
 4 Daniel went back to the hallway.
 5 Sandra moved to the garden.
 6 Where is Daniel?  hallway 4
 7 John moved to the office.
 8 Sandra journeyed to the bathroom.
 9 Where is Daniel?  hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom    8
</pre>
</div>

<!-- {% highlight text %}
 1 Mary moved to the bathroom.
 2 John went to the hallway.
 3 Where is Mary?    bathroom    1
 4 Daniel went back to the hallway.
 5 Sandra moved to the garden.
 6 Where is Daniel?  hallway 4
 7 John moved to the office.
 8 Sandra journeyed to the bathroom.
 9 Where is Daniel?  hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom    8
{% endhighlight %}
<br /><br /> -->

## Overview
<!-- The data released for task 1 consists of 400 instances (stories) divided into two equally sized mutually exclusive sets (train and test). Each instance contains 15 entries (10 statements and 5 questions). Task 1 deals exclusively with agents and their location. Across all the instances there are 4 agents and 6 locations. Each agent and location appears in both the train and test data.
 -->
<div class="fill-box" style="width: 60%; font-size: 24px;">
    Number of instances: 400
</div>
<div class="fill-box" style="width: 50%; float: right;  font-size: 24px; text-align: right;">
    Train instances: 200
</div>
<div class="fill-box" style="width: 70%; float: right;  font-size: 24px; text-align: right;">
    Test instances: 200
</div>
<div style="clear: both;"></div>

## Vocabulary
<div class="fill-box" style="width: 50%; font-size: 24px; text-align: right;">
    Size: 19 Words
</div>
<h2 class="centered">Agents</h2>
<ul class="stretch">
    <li style="width: 25%;">john</li>
    <li style="width: 25%;">mary</li>
    <li style="width: 25%;">sandra</li>
    <li style="width: 25%;">daniel</li>
</ul>

<h2 class="centered">Locations</h2>
<ul class="stretch">
    <li style="width: 16.66667%">bathroom</li>
    <li style="width: 16.66667%">bedroom</li>
    <li style="width: 16.66667%">office</li>
    <li style="width: 16.66667%">hallway</li>
    <li style="width: 16.66667%">kitchen</li>
    <li style="width: 16.66667%">garden</li>
</ul>

<h2 class="centered">Clause Templates</h2>

<ul class="stretch">
    <li style="width: 100%">AGENT went back to the LOCATION</li>
</ul>
<ul class="stretch">
    <li style="width: 100%">AGENT journeyed to the LOCATION</li>
</ul>
<ul class="stretch">
    <li style="width: 100%">AGENT travelled to the LOCATION</li>
</ul>
<ul class="stretch">
    <li style="width: 100%">AGENT went to the LOCATION</li>
</ul>
<ul class="stretch">
    <li style="width: 100%">AGENT moved to the LOCATION</li>
</ul>

<!-- <div class="fill-box">
    <ul style="list-style-type: none; margin: 0 0 0 25%;">
        <li style="font-family: monospace; font-size: 20px;">AGENT went back to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT journeyed to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT travelled to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT went to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT moved to the LOCATION</li>
    </ul>
</div> -->

<!-- <div class="grey-box" style="width: 60%; font-size: 24px;">
    Number of instances: 400
</div>
<div class="grey-box" style="width: 50%; float: right;  font-size: 24px; text-align: right;">
    Train instances: 200
</div>
<div class="grey-box" style="width: 50%; float: right;  font-size: 24px; text-align: right;">
    Test instances: 200
</div>
<div style="clear: both;"></div>

<div class="grey-box" style="width: 50%; margin-right: 25%;">
    <h2>Agents</h2>
    <ul style="list-style-type: none; margin-bottom: 0px;">
        <li style="font-family: monospace; font-size: 20px;">&nbsp;&nbsp;john</li>
        <li style="font-family: monospace; font-size: 20px;">&nbsp;&nbsp;mary</li>
        <li style="font-family: monospace; font-size: 20px;">sandra</li>
        <li style="font-family: monospace; font-size: 20px;">daniel</li>
    </ul>
</div>

<div class="grey-box" style="width: 75%; margin-left: 25%;">
    <h2 style="float: right;">Locations</h2>
    <div style="clear: both;"></div>
    <ul style="list-style-type: none; float: right; margin-bottom: 0px;">
        <li style="font-family: monospace; font-size: 20px;">bathroom</li>
        <li style="font-family: monospace; font-size: 20px;">&nbsp;bedroom</li>
        <li style="font-family: monospace; font-size: 20px;">&nbsp;&nbsp;office</li>
        <li style="font-family: monospace; font-size: 20px;">&nbsp;hallway</li>
        <li style="font-family: monospace; font-size: 20px;">&nbsp;kitchen</li>
        <li style="font-family: monospace; font-size: 20px;">&nbsp;&nbsp;garden</li>
    </ul>
    <div style="clear: both;"></div>
</div>

<div class="grey-box" style="width: 100%;">
    <h2 style="margin-left: 25%;">Statement Templates</h2>
    <ul style="list-style-type: none; width: 75%; margin: 0 25%;">
        <li style="font-family: monospace; font-size: 20px;">AGENT went back to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT journeyed to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT travelled to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT went to the LOCATION</li>
        <li style="font-family: monospace; font-size: 20px;">AGENT moved to the LOCATION</li>
    </ul>
</div> -->

## Number of Possible Clauses

$$
    n
    = \#\{\text{agents}\} \times \#\{\text{clauses}\} \times \#\{\text{locations}\} 
    = 4 \times 5 \times 6
    = 120.
$$

<!-- $$
    \begin{align*}
        n &= \#\{\text{agents}\} \times \#\{\text{statements}\} \times \#\{\text{locations}\} \\
          &= 4 \times 5 \times 6 \\
          &= 120.
    \end{align*}
$$ -->

## Instance Format
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">CLAUSE</li>
</ul>
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">CLAUSE</li>
</ul>
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">QUESTION</li>
</ul>
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">⋮</li>
</ul>
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">CLAUSE</li>
</ul>
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">CLAUSE</li>
</ul>
<ul class="stretch">
    <li style="width: 100%; text-align: left; padding: 0 0 0 10px">QUESTION</li>
</ul>


<!-- __4 Agents:__ `john, mary, sandra, daniel`

__6 Locations:__ `bathroom, bedroom, office, hallway, kitchen, garden`

After replacing each occurrence of an agent with the string `AGENT` and likewise for `LOCATION`, we're left with 5 unique statement templates.

__5 Statement Templates:__

1. `AGENT went back to the LOCATION`
2. `AGENT journeyed to the LOCATION`
3. `AGENT travelled to the LOCATION`
4. `AGENT went to the LOCATION`
5. `AGENT moved to the LOCATION`

Given these constraints there is a total of 120 possible unique statements.

$$
    \begin{align*}
        n &= \#\{\text{agents}\} \times \#\{\text{statements}\} \times \#\{\text{locations}\} \\
          &= 4 \times 5 \times 6 \\
          &= 120.
    \end{align*}
$$

Of these 120 possibilities all appear at least once in both the train and test data.

## A Closer Look at the Stories

Each story follows the same repeated format.

{% highlight text %}
1 STATEMENT 
2 STATEMENT 
3 QUESTION
⋮
12 QUESTION
13 STATEMENT 
14 STATEMENT 
15 QUESTION
{% endhighlight %}

When looking at the agent and locations appearing in each story, we finally find some variation. Below are plots illustrating the number of _unique_ agents and locations per story. -->

## Instance Composition
<img src="{{ site.baseurl }}/assets/deconstructing-babi-task-1/word-counts.png" />
<img src="{{ site.baseurl }}/assets/deconstructing-babi-task-1/unique-word-counts.png" />
<img src="{{ site.baseurl }}/assets/deconstructing-babi-task-1/agent-bars.png" />
<img src="{{ site.baseurl }}/assets/deconstructing-babi-task-1/location-bars.png" />

<!-- <img style="float: left;" src="{{ site.baseurl }}/assets/deconstructing-babi-task-1/agent-bars.png" />
<img style="float: left;" src="{{ site.baseurl }}/assets/deconstructing-babi-task-1/location-bars.png" />
<div style="clear: both;"></div> -->

<!-- I found this especially interesting. Unlike [Memory Networks](http://arxiv.org/abs/1410.3916), which have a number of memories equal to the number of statements in the story, my current work is focused on architectures for solving these sorts of problems using finite memory. Brazenly borrowing terminology from cognitive science "I'm exploring working memory solutions as opposed to long-term memory solutions." From this perspective the number of unique agents per story tells us something about the minimum capabilities required to solve the task. Specifically, a working memory solution will need to at minimum maintain information about the location of 4 agents simultaneously. -->

