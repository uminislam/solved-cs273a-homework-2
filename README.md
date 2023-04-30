Download Link: https://assignmentchef.com/product/solved-cs273a-homework-2
<br>
<h2>Problem 1: Linear Regression</h2>

For this problem we will explore linear regression, the creation of additional features, and cross-validation.

<table width="111">

 <tbody>

  <tr>

   <td width="111">data/curve80.txt</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Load the “” data set, and split it into 75% / 25% training/test. The first column</li>

</ol>

<table width="422">

 <tbody>

  <tr>

   <td width="63">data[:,0]</td>

   <td width="296">is the scalar feature (<em>x</em>) values; the second column</td>

   <td width="63">data[:,1]</td>

  </tr>

 </tbody>

</table>

is the target value <em>y </em>for each example. For consistency in our results, <strong>don’t </strong>reorder (shuffle) the data (they’re already in a random order), and use the first 75% of the data for training and the rest for testing:

<table width="591">

 <tbody>

  <tr>

   <td width="591">X = data[:,0]X    = np.atleast_2d(X).T # code expects shape (M,N) so make sure it’s 2-dimensionalY    = data[:,1]            # doesn’t matter for YXtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75) # split data set 75/25</td>

  </tr>

 </tbody>

</table>

1

2

3

4

Print the shapes of these four objects. <em>(5 points)</em>

<table width="90">

 <tbody>

  <tr>

   <td width="90">linearRegress</td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>Use the providedclass to create a linear regression predictor of <em>y </em>given <em>x</em>. You can plot the resulting function by simply evaluating the model at a large number of <em>x </em>values, xs :</li>

</ol>

<table width="591">

 <tbody>

  <tr>

   <td width="591">lr = ml.linear.linearRegress( Xtr, Ytr ) # create and train model xs = np.linspace(0,10,200)         # densely sample possible x-values xs = xs[:,np.newaxis]      # force “xs” to be an Mx1 matrix (expected by our code) ys = lr.predict( xs )              # make predictions at xs</td>

  </tr>

 </tbody>

</table>

1

2

3

4

<ul>

 <li>Plot the training data points along with your prediction function in a single plot. <em>(10 points)</em></li>

 <li>Print the linear regression coefficients ( theta ) and verify that they match your plot. <em>(5 points)</em></li>

 <li>What is the mean squared error of the predictions on the training and test data? <em>(10 points)</em></li>

</ul>

<ol start="3">

 <li>Try fitting <em>y </em>= <em>f </em>(<em>x</em>) using a polynomial function <em>f </em>(<em>x</em>) of increasing order. Do this by the trick of adding additional polynomial features before constructing and training the linear regression object. You can do this easily yourself; you can add a quadratic feature of Xtr with</li>

</ol>

<table width="591">

 <tbody>

  <tr>

   <td width="591">Xtr2 = np.zeros( (Xtr.shape[0],2) ) # create Mx2 array to store featuresXtr2[:,0] = Xtr[:,0]                                                 # place original “x” feature as X1Xtr2[:,1] = Xtr[:,0]**2                                        # place “x^2” feature as X2# Now, Xtr2 has two features about each data point: “x” and “x^2”</td>

  </tr>

 </tbody>

</table>

1

2

3

4

<table width="131">

 <tbody>

  <tr>

   <td width="131">ml.transforms.fpoly</td>

  </tr>

 </tbody>

</table>

(You can also add the all-ones constant feature in a similar way, but this is currently done automatically within the learner’s train function.) A function “” is also provided to more easily create such features. Note, though, that the resulting features may include extremely large values – if <em>x ≈ </em>10, then e.g., <em>x</em><sup>10 </sup>is extremely large. For this reason (as is often the case with features on very different scales) it’s a good idea to rescale the features; again, you can do this manually or use a provided rescale function:

<table width="591">

 <tbody>

  <tr>

   <td width="591"># Create polynomial features up to “degree”; don’t create constant feature# (the linear regression learner will add the constant feature automatically)XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)# Rescale the data matrix so that the features have similar ranges / varianceXtrP,params = ml.transforms.rescale(XtrP)# “params” returns the transformation parameters (shift &amp; scale)# Then we can train the model on the scaled feature matrix: lr = ml.linear.linearRegress( XtrP, Ytr )                # create and train model# Now, apply the same polynomial expansion &amp; scaling transformation to Xtest:XteP,_ = ml.transforms.rescale( ml.transforms.fpoly(Xte,degree,false), params)</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

<table width="117">

 <tbody>

  <tr>

   <td width="117">params = (mu,sig)</td>

  </tr>

 </tbody>

</table>

This snippet also shows a useful feature transformation framework – often we wish to apply some transformation to the features; in many cases the desired transformation depends on the data (such as rescaling the data to unit variance). Ideally, we should then be able to apply this same transform to new test data when it arrives, so that it will be treated in exactly the same way as the training data. “Feature transform” functions like rescale are written to output their settings, (here,, a tuple containing the mean and standard deviation used to shift and scale the data), so that they can be reused on subsequent data. You should create a function here that takes the degree as an argument, and returns the test set performance.

Train models of degree <em>d </em>= 1, 3, 5, 7, 10, 18 and:

<ul>

 <li>plot their learned prediction functions <em>f </em>(<em>x</em>) <em>(15 points)</em></li>

</ul>

<table width="57">

 <tbody>

  <tr>

   <td width="57">semilogy</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>plot their training and test errors on a log scale () as a function of the degree. <em>(10 points)</em></li>

 <li>What polynomial degree do you recommend? <em>(5 points)</em></li>

</ul>

<table width="43">

 <tbody>

  <tr>

   <td width="43">degree</td>

  </tr>

 </tbody>

</table>

<table width="36">

 <tbody>

  <tr>

   <td width="36">fpoly</td>

  </tr>

 </tbody>

</table>

For (a), remember that your learner has now been trained on the polynomially expanded features, and so is expectingfeatures (columns) to be input. So, don’t forget to also expand and scale the features of xs usingand rescale . You can do this manually as in the code snippet above, or you can think of this as a “feature transform” function Phi, eg.,

<table width="591">

 <tbody>

  <tr>

   <td width="591"># Define a function “Phi(X)” which outputs the expanded and scaled feature matrix: <strong>def </strong>Phi(X):<strong>return </strong>ml.transforms.rescale( ml.transforms.fpoly(X, degree,False), params)[0]# the parameters “degree” and “params” are memorized at the function definition# Now, Phi will do the required feature expansion and rescaling: YhatTrain = lr.predict( Phi(Xtr) )             # predict on training dataYhatTest = lr.predict( Phi(Xte) ) # predict on test data # etc.</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

Also, you may want to save the original axes of your plot and re-apply them to each subsequent plot for consistency. (Otherwise, high-degree polynomials may look “flat” due to some extremely large values.) You can do this as shown in Discussion 1 notebook by, for example:

<table width="591">

 <tbody>

  <tr>

   <td width="591"># Creating subplots with just one subplot so basically a single figure.fig, ax = plt.subplots(1, 1, figsize=(10, 8)) ax.plot(…) # Plot for each polynomial degree ax.plot(…) # like so ax.set_ylim(…, …) # Set the minimum and maximum limits plt.show()</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

<h2>Problem 2: Cross-validation</h2>

In the previous problem, you decided what degree of polynomial fit to use based on performance on some test data<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Let’s now imagine that you did not have access to the target values of the test data you held out in the previous problem, and wanted to decide on the best polynomial degree.

Of course, we could simply repeat the exercise, further splitting Xtr into a training and validation split, and then assessing performance on the validation data to decide on a degree. But when training is reasonably fast, it can be more effective to use cross-validation to estimate the optimal degree. Cross-validation works by creating many such training/validation splits, called folds, and using all of these splits to assess the “out-of-sample” (validation) performance by averaging them. You can do a 5-fold validation test, for example, by:

<table width="624">

 <tbody>

  <tr>

   <td width="624">nFolds = 5; <strong>for </strong>iFold <strong>in range</strong>(nFolds):Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold) # use ith block as validation learner = ml.linear.linearRegress(… # TODO: train on Xti, Yti, the data for this fold J[iFold] = … # TODO: now compute the MSE on Xvi, Yvi and save it# the overall estimated validation error is the average of the error on each fold <strong>print </strong>np.mean(J)</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

Using this technique on your training data Xtr from the previous problem, find the 5-fold cross-validation MSE of linear regression at the same degrees as before, <em>d </em>= 1, 3, 5, 7, 10, 18 (or more densely, if you prefer). Again, a function that has degree and number of folds as arguments, and returns cross-validation error, will be useful.

<table width="57">

 <tbody>

  <tr>

   <td width="57">semilogy</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Plot the <em>five-fold </em>cross-validation error and test error (with, as before) as a function of degree. <em>(10 points)</em></li>

 <li>How do the MSE estimates from five-fold cross-validation compare to the MSEs evaluated on the actual test data (Problem 1)? <em>(5 points)</em></li>

 <li>Which polynomial degree do you recommend based on five-fold cross-validation error? <em>(5 points)</em></li>

</ol>

<table width="57">

 <tbody>

  <tr>

   <td width="57">semilogy</td>

  </tr>

 </tbody>

</table>

<ol start="4">

 <li>For the degree that you picked in step 3, plot the cross-validation error as the number of folds is varied (nFolds = 2, 3, 4, 5, 6, 10, 12, 15), again with. What pattern do you observe, and how do you explain it? <em>(15 points)</em></li>

</ol>

<h2>Statement of Collaboration</h2>

It is <strong>mandatory </strong>to include a <em>Statement of Collaboration </em>in each submission, with respect to the guidelines below. Include the names of everyone involved in the discussions (especially in-person ones), and what was discussed.

All students are required to follow the academic honesty guidelines posted on the course website. For programming assignments, in particular, I encourage the students to organize (perhaps using Campuswire) to discuss the task descriptions, requirements, bugs in my code, and the relevant technical content <em>before </em>they start working on it. However, you should not discuss the specific solutions, and, as a guiding principle, you are not allowed to take anything written or drawn away from these discussions (i.e. no photographs of the blackboard,

written notes, referring to Campuswire, etc.). Especially <em>after </em>you have started working on the assignment, try to restrict the discussion to Campuswire as much as possible, so that there is no doubt as to the extent of your collaboration.