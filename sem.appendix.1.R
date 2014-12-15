# Matrix Algebra in R
# William Revelle
# Northwestern University 
# January 24, 2007

# Prepared as part of a course on Latent Variable Modeling, Winter, 2007 and as a supplement to the Guide to R for psychologists.

# email comments to: revelle@northwestern.edu

# A Matrix Algebra in R
# A.1 Vectors
# A.1.1 Vector multiplication
# A.1.2 Simple statistics using vectors
# A.1.3 Combining vectors
# A.2 Matrices
# A.2.1 Matrix addition 
# A.2.2 Matrix multiplication
# A.2.3 Finding and using the diagonal
# A.2.4 The Identity Matrix
# A.2.5 Matrix Inversion
# A.3 Matrix operations for data manipulation
# A.3.1 Matrix operations on the raw data
# A.3.2 Matrix operations on the correlation matrix
# A.3.3 Using matrices to find test reliability
# A.4 Multiple correlation
# A.4.1 Data level analyses
# A.4.2 Non optimal weights and the goodness of fit

# Appendix A:  Matrix Algebra in R

# Much of psychometrics in particular, and psychological data analysis in general consists of operations on vectors and matrices. This appendix offers a quick review of matrix operations with a particular emphasis upon how to do matrix operations in R. For more information on how to use R, consult the short guide to R for psychologists (at http://personality-project.org/r/r.guide.html) or the even shorter guide at personality-project.org/r/r.205.tutorial.html.

# A.1 Vectors

# A vector is a one dimensional array of n numbers. Basic operations on a vector are addition and subtraction. Multiplication is somewhat more complicated, for the order in which two vectors are multiplied changes the result. That is AB != BA.

# Consider V1 = the first 10 integers, and V2 = the next 10 integers:
  
V1 <- as.vector(seq(1, 10))
V1
V2 <- as.vector(seq(11, 20))
V2

# We can add a constant to each element in a vector

V4 <- V1 + 20
V4

# or we can add each element of the first vector to the corresponding element of the second vector

V3 <- V1 + V2
V3

# Strangely enough, a vector in R is dimensionless, but it has a length. If we want to multiply two vectors, we first need to think of the vector either as row or as a column. A column vector can be made into a row vector (and vice versa) by the transpose operation. While a vector has no dimensions, the transpose of a vector is two dimensional! It is a matrix with with 1 row and n columns. (Although the dim command will return no dimensions, in terms of multiplication, a vector is a matrix of n rows and 1 column.)

# Consider the following:

dim(V1)
length(V1)
dim(t(V1))
dim(t(t(V1)))
TV <- t(V1)
TV
t(TV)

# A.1.1 Vector multiplication

# Just as we can add a number to every element in a vector, so can we multiply a number (a “scaler”) by every element in a vector.

V2 <- 4 * V1
V2

# There are three types of multiplication of vectors in R. Simple multiplication (each term in one vector is multiplied by its corresponding term in the other vector), as well as the inner and outer products of two vectors.

# Simple multiplication requires that each vector be of the same length. Using the V1 and V2 vectors from before, we can find the 10 products of their elements:
  
V1
V2
V1*V2

# The “outer product” of a n x 1 element vector with a 1 x m element vector will result in a n x m element matrix. (The dimension of the resulting product is the outer dimensions of the two vectors in the multiplication). The vector multiply operator is %*%. In the following equation, the subscripts refer to the dimensions of the variable.

# Xnx1 ∗ Y1xm = (XY)nxm (A.1)

V1 <- seq(1, 10)
V2 <- seq(1, 4)
V1
V2
outer.prod <- V1 %*% t(V2)
outer.prod

# The outer product of the first ten integers is, of course, the multiplication table known to all elementary school students:
  
outer.prod <- V1 %*% t(V1)
outer.prod

# The “inner product” is perhaps a more useful operation, for it not only multiplies each corresponding element of two vectors, but also sums the resulting product:
  
inner.product = sum(V1i*V2i) (A.2)

V1 <- seq(1, 10)
V2 <- seq(11, 20)
V1
V2
in.prod <- t(V1) %*% V2
in.prod

# Note that the inner product of two vectors is of length = 1 but is a matrix with 1 row and 1 column. (This is the dimension of the inner dimensions (1) of the two vectors.)

# A.1.2 Simple statistics using vectors

# Although there are built in functions in R to do most of our statistics, it is useful to understand how these operations can be done using vector and matrix operations. Here we consider how to find the mean of a vector, remove it from all the numbers, and then find the average squared deviation from the mean (the variance).

# Consider the mean of all numbers in a vector. To find this we just need to add up the numbers (the inner product of the vector with a vector of 1s) and then divide by n (multiply by the scaler 1/n). First we create a vector of 1s by using the repeat operation. We then show three different equations for the mean.V, all of which are equivalent.

V <- V1
V
one <- rep(1, length(V))
one
sum.V <- t(one) %*% V
sum.V
mean.V <- sum.V * (1/length(V))
mean.V
mean.V <- t(one) %*% V * (1/length(V))
mean.V
mean.V <- t(one) %*% V/length(V)
mean.V

# The variance is the average squared deviation from the mean. To find the variance, we first find deviation scores by subtracting the mean from each value of the vector. Then, to find the sum of the squared deviations take the inner product of the result with itself. This Sum of Squares becomes a variance if we divide by the degrees of freedom (n - 1) to get an unbiased estimate of the population variance). First we find the mean centered vector:

V - mean.V

# And then we find the variance as the mean square by taking the inner product:
  
Var.V <- t(V - mean.V) %*% (V - mean.V) * (1/(length(V) - 1))
Var.V

# Compare these results with the more typical scale, mean and var operations:
  
scale(V, scale = FALSE)
mean(V)
var(V)

# A.1.3 Combining vectors

# We can form more complex data structures than vectors by combining the vectors, either by columns (cbind) or by rows (rbind). The resulting data structure is a matrix.

Xc <- cbind(V1, V2, V3)
Xc
Xr <- rbind(V1, V2, V3)
Xr
dim(Xc)
dim(Xr)

# A.2 Matrices

# A matrix is just a two dimensional (rectangular) organization of numbers. It is a vector of vectors. For data analysis, the typical data matrix is organized with columns representing different variables and rows containing the responses of a particular subject. Thus, a 10 x 4 data matrix (10 rows, 4 columns) would contain the data of 10 subjects on 4 different variables. Note that the matrix operation has taken the numbers 1 through 40 and organized them column wise. That is, a matrix is just a way (and a very convenient one at that) of organizing a vector.

# R provides numeric row and column names (e.g., [1,] is the first row, [,4] is the fourth column, but it is useful to label the rows and columns to make the rows (subjects) and columns (variables) distinction more obvious. [1]
                                         
Xij <- matrix(seq(1:40), ncol = 4)
rownames(Xij) <- paste("S", seq(1, dim(Xij)[1]), sep = "")
colnames(Xij) <- paste("V", seq(1, dim(Xij)[2]), sep = "")
Xij

# Just as the transpose of a vector makes a column vector into a row vector, so does the transpose of a matrix swap the rows for the columns. Note that now the subjects are columns and the variables are the rows.

t(Xij)

# A.2.1 Matrix addition

# The previous matrix is rather uninteresting, in that all the columns are simple products of the first column. A more typical matrix might be formed by sampling from the digits 0-9. For the purpose of this demonstration, we will set the random number seed to a memorable number so that it will yield the same answer each time.

set.seed(42)
Xij <- matrix(sample(seq(0, 9), 40, replace = TRUE), ncol = 4)
rownames(Xij) <- paste("S", seq(1, dim(Xij)[1]), sep = "")
colnames(Xij) <- paste("V", seq(1, dim(Xij)[2]), sep = "")
Xij

# Just as we could with vectors, we can add, subtract, multiply or divide the matrix by a scaler (a number with out a dimension).
                                         
Xij + 4
round((Xij + 4)/3, 2)

# We can also multiply each row (or column, depending upon order) by a vector.

V
Xij * V

# A.2.2 Matrix multiplication

# Matrix multiplication is a combination of multiplication and addition. For a matrix Xmxn of dimensions m x n and Ynxp of dimension n x p, the product, XYmxp is a m x p matrix where each element is the sum of the products of the rows of the first and the columns of the second. That is, the matrix XYmxp has elements xyij where each

# xyij = sum(xik*yjk) (A.3)

# Consider our matrix Xij with 10 rows of 4 columns. Call an individual element in this matrix xij. We can find the sums for each column of the matrix by multiplying the matrix by our “one” vector with Xij. That is, we can find sum(Xij) for the j columns, and then divide by the number (n) of rows. (Note that we can get the same result by finding colMeans(Xij).

# We can use the dim function to find out how many cases (the number of rows) or the number of variables (number of columns). dim has two elements: dim(Xij)[1] = number of rows, dim(Xij)[2] is the number of columns.
                                                              
dim(Xij)
n <- dim(Xij)[1]
n
one <- rep(1, n)
one
X.means <- t(one) %*% Xij/n
X.means

# A built in function to find the means of the columns is colMeans. (See rowMeans for the equivalent for rows.)

colMeans(Xij)

# Variances and covariances are measures of dispersion around the mean. We find these by first subtracting the means from all the observations. This means centered matrix is the original matrix minus a matrix of means. To make them have the same dimensions we premultiply the means vector by a vector of ones and subtract this from the data matrix.

X.diff <- Xij - one %*% X.means
X.diff

# To find the variance/covariance matrix, we can first find the the inner product of the means centered matrix X.diff = Xij - X.means t(Xij - X.means) with itself and divide by n-1. We can compare this result to the result of the cov function (the normal way to find covariances).

X.cov <- t(X.diff) %*% X.diff/(n - 1)
round(X.cov, 2)
round(cov(Xij), 2)

# A.2.3 Finding and using the diagonal

# Some operations need to find just the diagonal. For instance, the diagonal of the matrix X.cov (found above) contains the variances of the items. To extract just the diagonal, or create a matrix with a particular diagonal we use the diag command. We can convert the covariance matrix X.cov to a correlation matrix X.cor by pre and post multiplying the covariance matrix with a diagonal matrix containing the reciprocal of the standard deviations (square roots of the variances). Remember that the correlation, rxy, is merely the covariance(xy)/sqrt(VxVy). Compare this to the standard command for finding correlations cor.
                                                              
round(X.cov, 2)
round(diag(X.cov), 2)
sdi <- diag(1/sqrt(diag(X.cov)))
rownames(sdi) <- colnames(sdi) <- colnames(X.cov)
round(sdi, 2)
X.cor <- sdi %*% X.cov %*% sdi
rownames(X.cor) <- colnames(X.cor) <- colnames(X.cov)
round(X.cor, 2)
round(cor(Xij), 2)

# A.2.4 The Identity Matrix

# The identity matrix is merely that matrix, which when multiplied by another matrix, yields the other matrix. (The equivalent of 1 in normal arithmetic.) It is a diagonal matrix with 1 on the diagonal 

# I <- diag(1, nrow = dim(X.cov)[1], ncol = dim(X.cov)[2])

# A.2.5 Matrix Inversion

# The inverse of a square matrix is the matrix equivalent of dividing by that matrix. That is, either pre or post multiplying a matrix by its inverse yields the identity matrix. The inverse is particularly important in multiple regression, for it allows is to solve for the beta weights.

# Given the equation

# Y = bX + c (A.4)

# we can solve for b by multiplying both sides of the equation by

# X−1 or YX−1 = bXX−1 = b (A.5)

# We can find the inverse by using the solve function. To show that XX−1 = X−1X = I, we do the multiplication.

X.inv <- solve(X.cov)
X.inv
round(X.cov %*% X.inv, 2)
round(X.inv %*% X.cov, 2)

# There are multiple ways of finding the matrix inverse, solve is just one of them.

# A.3 Matrix operations for data manipulation

# Using the basic matrix operations of addition and multiplication allow for easy manipulation of data. In particular, finding subsets of data, scoring multiple scales for one set of items, or finding correlations and reliabilities of composite scales are all operations that are easy to do with matrix operations.

# In the next example we consider 5 extraversion items for 200 subjects collected as part of the Synthetic Aperture Personality Assessment project. The items are taken from the International Personality Item Pool (ipip.ori.org). The data are stored at the personality-project.org web site and may be retrieved in R. Because the first column of the data matrix is the subject identification number, we remove this before doing our calculations.

datafilename <- "http://personality-project.org/r/datasets/extraversion.items.txt"
items <- read.table(datafilename, header = TRUE)
items <- items[, -1]
dim(items)

# We first use functions from the psych package to describe these data both numerically and graphically. (The psych package may be downloaded from the personality-project.org web page as a source file.)

library(psych)
as.vector(installed.packages()[ , 1])
describe(items)
pairs.panels(items)

# Figure A.1: Scatter plot matrix (SPLOM) of 5 extraversion items (two reverse keyed) from the International Personality Item Pool.

# We can form two composite scales, one made up of the first 3 items, the other made up of the last 2 items. Note that the second (q1480) and fourth (q1180) are negatively correlated with the remaining 3 items. This implies that we should reverse these items before scoring.
                                                            
# To form the composite scales, reverse the items, and find the covariances and then correlations between the scales may be done by matrix operations on either the items or on the covariances between the items. In either case, we want to define a “keys” matrix describing which items to combine on which scale. The correlations are, of course, merely the covariances divided by the square root of the variances.
                                                              
# A.3.1 Matrix operations on the raw data
                                                              
keys <- matrix(c(1, -1, 1, 0, 0, 0, 0, 0, -1, 1), ncol = 2)
X <- as.matrix(items)
X.ij <- X %*% keys
n <- dim(X.ij)[1]
one <- rep(1, dim(X.ij)[1])
X.means <- t(one) %*% X.ij/n
X.cov <- t(X.ij - one %*% X.means) %*% (X.ij - one %*% X.means)/(n - 1)
round(X.cov, 2)
X.sd <- diag(1/sqrt(diag(X.cov)))
X.cor <- t(X.sd) %*% X.cov %*% (X.sd)
round(X.cor, 2)

# A.3.2 Matrix operations on the correlation matrix
                                                              
keys <- matrix(c(1, -1, 1, 0, 0, 0, 0, 0, -1, 1), ncol = 2)
X.cor <- cor(X)
round(X.cor, 2)
X.cov <- t(keys) %*% X.cor %*% keys
X.sd <- diag(1/sqrt(diag(X.cov)))
X.cor <- t(X.sd) %*% X.cov %*% (X.sd)
keys
round(X.cov, 2)
round(X.cor, 2)
                                                              
# A.3.3 Using matrices to find test reliability
                                                              
# The reliability of a test may be thought of as the correlation of the test with a test just like it. One conventional estimate of reliability, based upon the concepts from domain sampling theory, is coefficient alpha (alpha). For a test with just one factor, α is an estimate of the amount of the test variance due to that factor. However, if there are multiple factors in the test, α neither estimates how much the variance of the test is due to one, general factor, nor does it estimate the correlation of the test with another test just like it. (See Zinbarg et al., 2005 for a discussion of alternative estimates of reliability.)
                                                              
# Given either a covariance or correlation matrix of items, α may be found by simple matrix operations:
                                                                
# 1) V = the correlation or covariance matrix

# 2) Let Vt = the sum of all the items in the correlation matrix for that scale. 3) Let n = number of items in the scale

# 3) alpha = (Vt - diag(V))/Vt * n/(n-1)

# To demonstrate the use of matrices to find coefficient α, consider the five items measuring extraversion taken from the International Personality Item Pool. Two of the items need to be weighted negatively (reverse scored).

# Alpha may be found from either the correlation matrix (standardized alpha) or the covariance matrix (raw alpha). In the case of standardized alpha, the diag(V) is the same as the number of items. Using a key matrix, we can find the reliability of 3 different scales, the first is made up of the first 3 items, the second of the last 2, and the third is made up of all the items.

datafilename <- "http://personality-project.org/r/datasets/extraversion.items.txt" 
items <- read.table(datafilename, header = TRUE)
items <- items[, -1]
key <- matrix(c(1, -1, 1, 0, 0, 0, 0, 0, -1, 1, 1, -1, 1, -1, 1), ncol = 3)
key
raw.r <- cor(items)
V <- t(key) %*% raw.r %*% key
rownames(V) <- colnames(V) <- c("V1-3", "V4-5", "V1-5")
round(V, 2)
n <- diag(t(key) %*% key)
alpha <- (diag(V) - n)/(diag(V)) * (n/(n - 1))
round(alpha, 2)

# A.4 Multiple correlation

# Given a set of n predictors of a criterion variable, what is the optimal weighting of the n predictors? This is, of course, the problem of multiple correlation or multiple regression. Although we would normally use the linear model (lm) function to solve this problem, we can also do it from the raw data or from a matrix of covariances or correlations by using matrix operations and the solve function.

# Consider the data set, X, created in section A.2.1. If we want to predict V4 as a function of the first three variables, we can do so three different ways, using the raw data, using deviation scores of the raw data, or with the correlation matrix of the data.

# For simplicity, lets relabel V4 to be Y and V1...V3 to be X1...X3 and then define X as the first three columns and Y as the last column:

set.seed(42)
Xij <- matrix(sample(seq(0, 9), 40, replace = TRUE), ncol = 4)
rownames(Xij) <- paste("S", seq(1, dim(Xij)[1]), sep = "")
colnames(Xij) <- paste("V", seq(1, dim(Xij)[2]), sep = "")
X <- Xij[ , 1:3]
colnames(X) <- c("X1", "X2", "X3")
X
Y <- Xij[, 4]
Y

# A.4.1 Data level analyses

# At the data level, we can work with the raw data matrix X, or convert these to deviation scores (X.dev) by subtracting the means from all elements of X. At the raw data level we have

# Yˆmx1 = Xmxnβ1nx1 + ε1mx1 (A.6) 

# and we can solve for nβ1 by pre multiplying by nXm′ (thus making the matrix on the right side of the equation into a square matrix so that we can multiply through by an inverse.)

# X′nxmYˆmx1 = X′nxmXmxnβnx1 + εmx1 (A.7)

# and then solving for beta by pre multiplying both sides of the equation by (XX′)−1

# β = (XX′)−1X′Y (A.8) 

# These beta weights will be the weights with no intercept. Compare this solution to the one using the lm function with the intercept removed:
  
beta <- solve(t(X) %*% X) %*% (t(X) %*% Y)
round(beta, 2)
lm(Y ~ -1 + X)

# If we want to find the intercept as well, we can add a column of 1s to the X matrix.

one <- rep(1, dim(X)[1])
X <- cbind(one, X)
X
beta <- solve(t(X) %*% X) %*% (t(X) %*% Y)
round(beta, 2)
lm(Y ~ X)

# We can do the same analysis with deviation scores. Let X.dev be a matrix of deviation scores, then can write the equation

# Yˆ = Xβ + ε (A.9) 

# and solve for

# β = (X.devX.dev′)−1X.dev′Y (A.10)

# (We don’t need to worry about the sample size here because n cancels out of the equation).

# At the structure level, the covariance matrix = XX’/(n-1) and X’Y/(n-1) may be replaced by correlation matrices by pre and post multiplying by a diagonal matrix of 1/sds) with rxy and we then solve the equation

# β = R−1rxy (A.11)

# Consider the set of 3 variables with intercorrelations (R)

#     x1   x2   x3
# x1 1.00 0.56 0.48
# x2 0.56 1.00 0.42
# x3 0.48 0.42 1.00

# and correlations of x with y (rxy)

#    x1  x2   x3
# y 0.4 0.35 0.3

# From the correlation matrix, we can use the solve function to find the optimal beta weights.

R <- matrix(c(1, 0.56, 0.48, 0.56, 1, 0.42, 0.48, 0.42, 1), ncol = 3)
rxy <- matrix(c(0.4, 0.35, 0.3), ncol = 1)
colnames(R) <- rownames(R) <- c("x1", "x2", "x3")
rownames(rxy) <- c("x1", "x2", "x3")
colnames(rxy) <- "y"
beta <- solve(R, rxy)
round(beta, 2)

# A.4.2 Non optimal weights and the goodness of fit

# Although the beta weights are optimal given the data, it is well known (e.g., the robust beauty of linear models by Robyn Dawes) that if the predictors are all adequate and in the same direction, the error in prediction of Y is rather insensitive to the weights that are used. This can be shown graphically by comparing varying the weights of x1 and x2 relative to x3 and then finding the error in prediction. Note that the surface is relatively flat at its minimum.

# We show this for several different values of the rxy and R matrices by first defining two functions (f and g) and then applying these functions with different values of R and rxy. The first, f, finds the multiple r for values of bx1/bx3 and bx2/bx3 for any value or set of values given by the second function. ranging from low to high and then find the error variance (1 − r2) for each case.

f <- function(x, y) {
  xy <- diag(c(x, y, 1))
  c <- rxy %*% xy
  d <- xy %*% R %*% xy
  cd <- sum(c)/sqrt(sum(d))
  return(cd)
}

g <- function(rxy, R, n = 60, low = -2, high = 4, ...) {
  op <- par(bg = "white")
  x <- seq(low, high, length = n)
  y<-x
  z <- outer(x, y)
  for(i in 1:n) {
    for (j in 1:n) {
      r <- f(x[i], y[j])
      z[i, j] <- 1 - r^2
    }
  }
  persp(x, y, z, theta = 40, phi = 30, expand = 0.5, col = "lightblue",
        ltheta = 120, shade = 0.75,
        ticktype = "detailed", zlim = c(0.5, 1), xlab = "x1/x3",
        ylab = "x2/x3", zlab = "Error")
  zmin <- which.min(z)
  ymin <- trunc(zmin/n)
  xmin <-zmin - ymin*n
  xval <- x[xmin + 1]
  yval <- y[trunc(ymin) + 1]
  title(paste("Error as function of relative weights  min values at x1/x3 = ",
              round(xval, 1), " x2/x3 = ", round(yval, 1)))
}

R <- matrix(c(1, 0.56, 0.48, 0.56, 1, 0.42, 0.48, 0.42, 1), ncol = 3) 
rxy <- matrix(c(0.4, 0.35, 0.3), nrow = 1)
colnames(R) <- rownames(R) <- c("x1", "x2", "x3")
colnames(rxy) <- c("x1", "x2", "x3")
rownames(rxy) <- "y"

# Figure A.2 shows the residual error variance as a function of relative weights of bx1/bx3 and bx2/bx3 for a set of correlated predictors.

R
rxy
g(R, rxy)

# Figure A.2: Squared Error as function of misspecified beta weights. Although the optimal weights minimize the squared error, a substantial variation in the weights does not lead to very large change in the squared error.

# Figure A.3: With orthogonal predictors, the squared error surface is somewhat better defined, although still large changes in beta weights do not lead to large changes in squared error.

# With independent predictors, the response surface is somewhat steeper (Figure A.3), but still the squared error function does not change vary rapidly as the beta weights change. This phenomenon has been well discussed by Robyn Dawes and his colleagues.

#    x1 x2 x3
# x1  1  0  0
# x2  0  1  0
# x3  0  0  1

#    x1   x2  x3
# y 0.4 0.35 0.3


[1] Although many think of matrices as developed in the 17th century, O’Conner and Robertson discuss the history of matrix algebra back to the Babylonians (http://www-history.mcs.st-andrews.ac.uk/history/HistTopics/Matrices_and_determinants.html)