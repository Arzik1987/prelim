#' PRIM returning peeling trajectory
#'
#' The function applies PRIM to train data and evaluates its quality on test data
#'
#' @param dtrain list, containing training data. The first element contains matrix/data frame of real attribute values.
#' the second element contains vector of labels 0/1.
#' @param dtest list, containing test data. Structured in the same way as \code{dtrain}. If NULL, the
#' quality metrics on test data are not computed
#' @param deval list, containing data for evaluation. Structured in the same way as \code{dtrain}.
#' By default coincides with \code{dtrain}
#' @param box matrix of real. Initial hyperbox, covering data
#' @param minpts integer. Minimal number of points in the box for PRIM to continue peeling
#' @param max.peels integer. Maximum length of the peeling trajectory (number of boxes)
#' @param peel.alpha a set of real. The peeling parameter(s) of PRIM from the interval (0,1).
#' If a vector, the value is selected with \code{\link{select.alpha}} algorithm
#' @param pasting logical. If  TRUE, pasting is used on each box forming the peeling trajectory
#' @param paste.alpha real. The pasting parameter of PRIM from the interval (0,1)
#' @param threshold real. If precision of the current box on \code{train}
#' is greater or equal \code{threshold}, PRIM stops peeling
#' @param seed seed for reproducibility of hyperparameter optimization procedure.
#' Default is 2020. Set NULL for not using
#'
#' @keywords models, multivariate
#'
#' @references Friedman, J.H. and Fisher, N.I. 1999. Bump hunting in high-dimensional data.
#' Statistics and Computing. 9, 2 (1999), 123-143.
#'
#' @return list.
#' \itemize{
#' \item \code{pr.test} matrix with coverage (recall) in the first column and
#' density (precision) in the second column, evaluated on \code{dtest}
#' \item \code{pr.eval} matrix with coverage (recall) in the first column and
#' density (precision) in the second column, evaluated on \code{deval}
#' \item \code{boxes} list of matrices defining boxes constituting peeling trajectory
#' \item \code{peel.alpha} integer; the value of \code{peel.alpha} parameter used
#' }
#'
#' @importFrom stats quantile
#'
#' @seealso \code{\link{rf.prim}},
#' \code{\link{bumping.prim}}
#'
#' @export
#'
#' @examples
#'
#' dtrain <- dtest <- list()
#' dtest[[1]] <- dsgc_sym[1001:10000, 1:12]
#' dtest[[2]] <- dsgc_sym[1001:10000, 13]
#' dtrain[[1]] <- dsgc_sym[1:500, 1:12]
#' dtrain[[2]] <- dsgc_sym[1:500, 13]
#' box <- matrix(c(0.5,0.5,0.5,0.5,1,1,1,1,0.05,0.05,0.05,0.05,
#' 5,5,5,5,4,4,4,4,1,1,1,1), nrow = 2, byrow = TRUE)
#'
#' res1 <- norm.prim(dtrain = dtrain, dtest = dtest, box = box)
#' res2 <- norm.prim(dtrain = dtrain, dtest = dtest, box = box, pasting = TRUE)
#' res3 <- norm.prim(dtrain = dtrain, dtest = dtest, box = box,
#' peel.alpha = c(0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.2))
#'
#' plot(res3[[1]], col = "green", type = "l",
#' xlab = "recall", ylab = "precision")
#' lines(res2[[1]], col = "blue")
#' lines(res1[[1]], col = "brown")
#' legend("bottomleft", legend = c("prim.cv", "prim.pasting", "prim"),
#' col = c("green", "blue", "brown"), lty = c(1, 1, 1))


norm.prim <- function(dtrain, dtest = NULL, deval = dtrain, box, minpts = 20, max.peels = 999,
                     peel.alpha = 0.05, pasting = FALSE, paste.alpha = 0.01, threshold = 1,
                     seed = 2020){

  time1 = Sys.time()

  if(length(peel.alpha) > 1){
    peel.alpha <- select.alpha(dtrain = dtrain, box = box, minpts = minpts,
                               max.peels = max.peels, peel.alpha = peel.alpha,
                               threshold = threshold, seed = seed)
  }

  peel <- function(){

    hgh <- -Inf
    bnd <- -Inf
    vol.red <- 1

    for(i in 1:ncol(x)){
      bound <- quantile(x[, i], peel.alpha, type = 8)
      vol.r <- (bound - box[1, i])/(box[2, i] - box[1, i])
      retain <- (x[, i] > bound)                                   # this inequality implicitly assumes low (< peel.alpha) share of duplicates for each value
      if(sum(retain)){
        tar <- sum(y[retain])/sum(retain)
        if(tar > hgh | (tar == hgh & vol.r < vol.red)){
          hgh <- tar
          vol.red <- vol.r
          inds <- retain
          rn = 1
          cn = i
          bnd = bound
        }
      }
      bound <- quantile(x[, i], 1 - peel.alpha, type = 8)
      vol.r <- (box[2, i] - bound)/(box[2, i] - box[1, i])
      retain <- (x[, i] < bound)
      if(sum(retain)){
        tar <- sum(y[retain])/sum(retain)
        if(tar > hgh | (tar == hgh & vol.r < vol.red)){
          vol.red <- vol.r
          hgh <- tar
          inds <- retain
          rn = 2
          cn = i
          bnd = bound
        }
      }
    }
    if(hgh == -Inf){
      continue.peeling <<- FALSE
    } else {
      x <<- x[inds,, drop = FALSE]
      y <<- y[inds]
      box[rn, cn] <<- bnd
      continue.peeling <<- (hgh < threshold)
    }
  }

  qual.pr <- function(d, box.p){
    Np = sum(d[[2]])
    retain <- rep(TRUE, length(d[[2]]))
    for(i in 1:ncol(d[[1]])){
      retain <- retain & d[[1]][, i] > box.p[1, i]
      retain <- retain & d[[1]][, i] < box.p[2, i]
    }
    n = length(d[[2]][retain])
    np = sum(d[[2]][retain])
    rec <- np/Np
    pr <- np/n
    c(rec, pr, sum(retain))
  }


  x <- dtrain[[1]]
  y <- dtrain[[2]]
  continue.peeling <- TRUE
  neval <- nrow(deval[[1]])

  i = 0
  boxes <- list()
  box.p <- box
  q <- qtest <- matrix(ncol = 2, nrow = 0)

  while(length(y) >= minpts & neval >= minpts & continue.peeling & i <= max.peels){
    temp <- qual.pr(deval, box.p)
    neval <- temp[3]
    q <- rbind(q, temp[1:2])
    i <- i + 1
    boxes <- append(boxes, list(box.p))
    peel()
    box.p <- box

    # pasting step (use prim package)
    if(pasting){
      res.p <- list(x = x, y = y, box = box)
      while (!is.null(res.p)){
        box.p <- res.p$box
        res.p <- prim:::paste.one(x = res.p$x, y = res.p$y, box = res.p$box,
                                      x.init = dtrain[[1]], y.init = dtrain[[2]], paste.alpha = paste.alpha,
                                      mass.min = 0, threshold = 0, d = ncol(dtrain[[1]]), n = 1,
                                      y.fun = mean, verbose = FALSE)
      }
    }
    # end of pasting
  }

  ret <- which(q[, 2] == max(q[, 2]))[1]
  q <- matrix(q[1:ret,], ncol = 2)
  boxes <- boxes[1:ret]

  time.train = Sys.time() - time1

  if(!is.null(dtest)){
    for(i in boxes){
      qtest <- rbind(qtest, qual.pr(dtest, i)[1:2])
    }
  }

  return(list(pr.test = qtest, pr.eval = q, boxes = boxes,
              peel.alpha = peel.alpha, time.train = time.train))
}



dx = matrix(runif(400000), ncol = 4)
dy = (apply((dx > 0.3), 1, sum) == 4) - 0
box = matrix(c(rep(0, 4), rep(1, 4)), ncol = 4, byrow = TRUE)
dtrain <- list(dx, dy)
start = Sys.time()
bp = norm.prim(dtrain = dtrain, box = box)
end = Sys.time()
print(end - start) # ~1.1s

