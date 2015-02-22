<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">

<head>

<meta charset="utf-8">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="pandoc" />



<title>Machine learning On lifting technique</title>

<style type="text/css">
  pre:not([class]) {
    background-color: white;
  }
</style>
<script type="text/javascript">
if (window.hljs && document.readyState && document.readyState === "complete") {
   window.setTimeout(function() {
      hljs.initHighlighting();
   }, 0);
}
</script>



</head>

<body>

<style type="text/css">
.main-container {
  max-width: 940px;
  margin-left: auto;
  margin-right: auto;
}
</style>
<div class="container-fluid main-container">


<div id="header">
<h1 class="title">Machine learning On lifting technique</h1>
</div>


<div id="summary" class="section level1">
<h1>Summary</h1>
<p>We attempt to form a model which can predict the type of lifting technique used based on sensor data. It is found that a random forest model has an average error of 0.4% when estimated with k folds testing. The final model predicted all observations in the data set correctly.</p>
</div>
<div id="introduction" class="section level1">
<h1>Introduction</h1>
<p>We would like to be able to predict whether, and in which way, an activity is being performed with incorrect technique. We attempt to do this using machine learning on a data set gathered from sensors positioned on the body of several different people as they performed lifts in both correct, and incorrect manners (recorded in the classe variable).</p>
</div>
<div id="cleaning-data" class="section level1">
<h1>Cleaning data</h1>
<p>We first import the training data, and load all required packages:</p>
<pre class="r"><code>training&lt;-read.csv(&quot;http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv&quot;,na.strings=c(&quot;&quot;,&quot;NA&quot;))

library(randomForest)</code></pre>
<pre><code>## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.</code></pre>
<pre class="r"><code>library(gbm)</code></pre>
<pre><code>## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1</code></pre>
<p>We would now like to clean the data, so that only variables which are not mainly NA’s are included:</p>
<pre class="r"><code>removeMainlyNA &lt;- function(inputData){
  inputNames&lt;-names(inputData)
  outputNames&lt;-NULL
  length&lt;-length(training[,1])
  for (name in inputNames){
    if (sum(is.na(inputData[name]))/length &lt; 0.5){
      outputNames&lt;-c(outputNames,name)
    }
  }
  inputData[,outputNames]
}

cleanData&lt;-removeMainlyNA(training)

names(cleanData)</code></pre>
<pre><code>##  [1] &quot;X&quot;                    &quot;user_name&quot;            &quot;raw_timestamp_part_1&quot;
##  [4] &quot;raw_timestamp_part_2&quot; &quot;cvtd_timestamp&quot;       &quot;new_window&quot;          
##  [7] &quot;num_window&quot;           &quot;roll_belt&quot;            &quot;pitch_belt&quot;          
## [10] &quot;yaw_belt&quot;             &quot;total_accel_belt&quot;     &quot;gyros_belt_x&quot;        
## [13] &quot;gyros_belt_y&quot;         &quot;gyros_belt_z&quot;         &quot;accel_belt_x&quot;        
## [16] &quot;accel_belt_y&quot;         &quot;accel_belt_z&quot;         &quot;magnet_belt_x&quot;       
## [19] &quot;magnet_belt_y&quot;        &quot;magnet_belt_z&quot;        &quot;roll_arm&quot;            
## [22] &quot;pitch_arm&quot;            &quot;yaw_arm&quot;              &quot;total_accel_arm&quot;     
## [25] &quot;gyros_arm_x&quot;          &quot;gyros_arm_y&quot;          &quot;gyros_arm_z&quot;         
## [28] &quot;accel_arm_x&quot;          &quot;accel_arm_y&quot;          &quot;accel_arm_z&quot;         
## [31] &quot;magnet_arm_x&quot;         &quot;magnet_arm_y&quot;         &quot;magnet_arm_z&quot;        
## [34] &quot;roll_dumbbell&quot;        &quot;pitch_dumbbell&quot;       &quot;yaw_dumbbell&quot;        
## [37] &quot;total_accel_dumbbell&quot; &quot;gyros_dumbbell_x&quot;     &quot;gyros_dumbbell_y&quot;    
## [40] &quot;gyros_dumbbell_z&quot;     &quot;accel_dumbbell_x&quot;     &quot;accel_dumbbell_y&quot;    
## [43] &quot;accel_dumbbell_z&quot;     &quot;magnet_dumbbell_x&quot;    &quot;magnet_dumbbell_y&quot;   
## [46] &quot;magnet_dumbbell_z&quot;    &quot;roll_forearm&quot;         &quot;pitch_forearm&quot;       
## [49] &quot;yaw_forearm&quot;          &quot;total_accel_forearm&quot;  &quot;gyros_forearm_x&quot;     
## [52] &quot;gyros_forearm_y&quot;      &quot;gyros_forearm_z&quot;      &quot;accel_forearm_x&quot;     
## [55] &quot;accel_forearm_y&quot;      &quot;accel_forearm_z&quot;      &quot;magnet_forearm_x&quot;    
## [58] &quot;magnet_forearm_y&quot;     &quot;magnet_forearm_z&quot;     &quot;classe&quot;</code></pre>
<p>We can see that the first 7 variables remaining do not appear to be relevant, as they contain only timestamp data, user names, and the index of the observation.</p>
<pre class="r"><code>cleanData&lt;-cleanData[,8:60]</code></pre>
</div>
<div id="building-model" class="section level1">
<h1>Building Model</h1>
<div id="cross-validation" class="section level2">
<h2>Cross validation</h2>
<pre class="r"><code>crossValidate &lt;- function(inputData,repeats,k){
  length&lt;-length(inputData[,1])
  sampleSize&lt;-round(length/k)
  accuracy&lt;-NULL
  while (repeats != 0){
    random&lt;-sample(length,size=length)
    toDo&lt;-k
    location&lt;-1
    while (toDo != 0){
      thing&lt;-random[location:(min(location+sampleSize,length))]
      train &lt;- inputData[-thing,]
      test &lt;- inputData[thing,]
      model&lt;-randomForest(classe~., train)
      predictions &lt;- predict(model,test)
      result&lt;-confusionMatrix(predictions,test$classe)
      accuracy&lt;-c(accuracy,result$overall[1])
      toDo&lt;-toDo-1
      location&lt;-location+sampleSize
    }
    
    repeats &lt;- repeats - 1
  }
  mean(accuracy)
}</code></pre>
<p>For a balance between accuracy and speed:</p>
<pre class="r"><code>accuracy&lt;-crossValidate(cleanData,3,5)</code></pre>
<p>This gives a very high accuracy of 99.6%</p>
</div>
<div id="training" class="section level2">
<h2>Training</h2>
<p>now we can run some a training function on the entire data set, basing “classe” on all other available variables:</p>
<pre class="r"><code>model&lt;-randomForest(classe~., cleanData)

model</code></pre>
<pre><code>## 
## Call:
##  randomForest(formula = classe ~ ., data = cleanData) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.29%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5578    1    0    0    1 0.0003584229
## B   10 3783    4    0    0 0.0036871214
## C    0    8 3412    2    0 0.0029222677
## D    0    0   23 3192    1 0.0074626866
## E    0    0    2    5 3600 0.0019406709</code></pre>
</div>
<div id="expected-out-of-sample-error" class="section level2">
<h2>Expected Out of sample error</h2>
<p>We found that the random forest training method had an average accuracy of 99.6%, and error of 0.4%. The subsets of the data which we trained and tested on were selected randomly, so we would expect the error on the final test set to be reasonably close to this, but a little higher as over fitting is difficult to avoid and has almost certainly occurred.</p>
<p>This is a very low error, so we are happy to use this as our final model</p>
</div>
</div>
<div id="testing" class="section level1">
<h1>Testing</h1>
<pre class="r"><code>test&lt;-read.csv(&quot;http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv&quot;,na.strings=c(&quot;&quot;,&quot;NA&quot;))</code></pre>
<p>Now we can predict the test set using the model:</p>
<pre class="r"><code>results&lt;-predict(model,test)
results</code></pre>
<pre><code>##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E</code></pre>
<div id="actual-error" class="section level2">
<h2>Actual error</h2>
<p>All predicted results were found to be correct. This gives a final error of 0%. It is likely that if given a larger set of test data we would find a higher error.</p>
</div>
</div>


</div>

<script>

// add bootstrap table styles to pandoc tables
$(document).ready(function () {
  $('tr.header').parent('thead').parent('table').addClass('table table-condensed');
});

</script>

<!-- dynamically load mathjax for compatibility with self-contained -->
<script>
  (function () {
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.src  = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
    document.getElementsByTagName("head")[0].appendChild(script);
  })();
</script>

</body>
</html>
