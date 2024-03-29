### Predict method for LDA objects
#' Get predictions from a Latent Dirichlet Allocation model
#' @description Obtains predictions of topics for new documents from a fitted LDA model
#' @param object a fitted object of class \code{tidylda}
#' @param new_data a DTM or TCM of class \code{dgCMatrix} or a numeric vector
#' @param type one of "prob", "class", or "distribution". Defaults to "prob".
#' @param method one of either "gibbs" or "dot". If "gibbs" Gibbs sampling is used
#'        and \code{iterations} must be specified.
#' @param iterations If \code{method = "gibbs"}, an integer number of iterations
#'        for the Gibbs sampler to run. A future version may include automatic stopping criteria.
#' @param burnin If \code{method = "gibbs"}, an integer number of burnin iterations.
#'        If \code{burnin} is greater than -1, the entries of the resulting "theta" matrix
#'        are an average over all iterations greater than \code{burnin}.
#'        Behavior is the same as documented in \code{\link[tidylda]{tidylda}}.
#' @param no_common_tokens behavior when encountering documents that have no tokens
#'        in common with the model. Options are "\code{default}", "\code{zero}",
#'        or "\code{uniform}". See 'details', below for explanation of behavior. 
#' @param times Integer, number of samples to draw if \code{type = "distribution"}.
#'   Ignored if \code{type} is "class" or "prob". Defaults to 100.
#' @param threads Number of parallel threads, defaults to 1. Note: currently
#'   ignored; only single-threaded prediction is implemented.
#' @param verbose Logical. Do you want to print a progress bar out to the console?
#'        Only active if \code{method = "gibbs"}. Defaults to \code{TRUE}.
#' @param ... Additional arguments, currently unused
#' @return \code{type} gives different outputs depending on whether the user selects
#'   "prob", "class", or "distribution". If "prob", the default, returns a
#'   a "theta" matrix with one row per document and one column per topic. If
#'   "class", returns a vector with the topic index of the most likely topic in
#'   each document. If "distribution", returns a tibble with one row per
#'   parameter per sample. Number of samples is set by the \code{times} argument.
#' @details
#'   If \code{predict.tidylda} encounters documents that have no tokens in common
#'   with the model in \code{object} it will engage in one of three behaviors based
#'   on the setting of \code{no_common_tokens}.
#'   
#'   \code{default} (the default) sets all topics to 0 for offending documents. This 
#'   enables continued computations downstream in a way that \code{NA} would not.
#'   However, if \code{no_common_tokens == "default"}, then \code{predict.tidylda}
#'   will emit a warning for every such document it encounters.
#'   
#'   \code{zero} has the same behavior as \code{default} but it emits a message
#'   instead of a warning.
#'   
#'   \code{uniform} sets all topics to 1/k for every topic for offending documents.
#'   it does not emit a warning or message.
#' @examples
#' \donttest{
#' # load some data
#' data(nih_sample_dtm)
#'
#' # fit a model
#' set.seed(12345)
#'
#' m <- tidylda(
#'   data = nih_sample_dtm[1:20, ], k = 5,
#'   iterations = 200, burnin = 175
#' )
#'
#' str(m)
#'
#' # predict on held-out documents using gibbs sampling "fold in"
#' p1 <- predict(m, nih_sample_dtm[21:100, ],
#'   method = "gibbs",
#'   iterations = 200, burnin = 175
#' )
#'
#' # predict on held-out documents using the dot product
#' p2 <- predict(m, nih_sample_dtm[21:100, ], method = "dot")
#'
#' # compare the methods
#' barplot(rbind(p1[1, ], p2[1, ]), beside = TRUE, col = c("red", "blue"))
#' 
#' # predict classes on held out documents
#' p3 <- predict(m, nih_sample_dtm[21:100, ],
#'   method = "gibbs",
#'   type = "class",
#'   iterations = 100, burnin = 75
#' )
#' 
#' # predict distribution on held out documents
#' p4 <- predict(m, nih_sample_dtm[21:100, ],
#'   method = "gibbs",
#'   type = "distribution",
#'   iterations = 100, burnin = 75,
#'   times = 10
#' )
#' }
#' @export
predict.tidylda <- function(
  object, 
  new_data, 
  type = c("prob", "class", "distribution"),
  method = c("gibbs", "dot"),
  iterations = NULL, 
  burnin = -1, 
  no_common_tokens = c("default", "zero", "uniform"),
  times = 100,
  threads = 1,
  verbose = TRUE,
  ...
){
  
  ### Check inputs ----
  if (! type[1] %in% c("prob", "class", "distribution")) {
    stop("type must be one of 'prob', 'class', or 'distribution'. I see type = ", type[1])
  }
  
  if (type[1] == "distribution") {
    if (is.na(times[1]) | is.infinite(times[1])) {
      stop("times must be a number 1 or greater. I see times = ", times)
    }
    
    if (! is.numeric(times[1])) {
      stop("times must be a number 1 or greater. I see times = ", times)
    }
    
    if (times[1] < 1) {
      stop("times must be a number 1 or greater. I see times = ", times)
    }
    
    times <- round(times[1])
  }
  
  if (method[1] == "gibbs") {
    if (is.null(iterations)) {
      stop("when using method 'gibbs' iterations must be specified.")
    }

    if (burnin >= iterations) {
      stop("burnin must be less than iterations")
    }
  }

  # handle dtm
  new_data <- convert_dtm(dtm = new_data)

  if (sum(c("gibbs", "dot") %in% method) == 0) {
    stop("method must be one of 'gibbs' or 'dot'")
  }

  dtm_new_data <- new_data

  if (sum(no_common_tokens %in% c("default", "zero", "uniform")) <= 0) {
    stop(
      "no_common_tokens must be one of 'default', 'zero', or 'uniform'."
    )
  }
  
  # check threads against nrow(dtm_new_data)
  # only matters if method = "gibbs"
  if (threads > 1)
    threads <- as.integer(max(floor(threads), 1)) # prevent any decimal inputs
  
  if (method[1] == "gibbs" & threads > nrow(dtm_new_data)) {
    message("User-supplied 'threads' argument greater than number of documents.\n",
            "Setting threads equal to number of documents.")
    threads <- as.integer(nrow(dtm_new_data))
  }
  
  ### Align vocabulary ----
  # this is fancy because of how we do indexing in gibbs sampling
  vocab_original <- colnames(object$beta) # tokens in training set

  vocab_intersect <- intersect(vocab_original, colnames(dtm_new_data))

  vocab_add <- setdiff(vocab_original, vocab_intersect)

  add_mat <- Matrix::Matrix(0, nrow = nrow(dtm_new_data), ncol = length(vocab_add))

  colnames(add_mat) <- vocab_add

  dtm_new_data <- Matrix::cbind2(dtm_new_data, add_mat)

  if (nrow(dtm_new_data) == 1) {
    dtm_new_data <- Matrix::Matrix(dtm_new_data[, vocab_original], nrow = 1, sparse = TRUE)

    colnames(dtm_new_data) <- vocab_original

    rownames(dtm_new_data) <- 1
  } else {
    dtm_new_data <- dtm_new_data[, vocab_original]
  }

  ### Get predictions ----

  if (method[1] == "dot") { # dot product method

    result <- dtm_new_data[, vocab_original]

    # handle differently if one row
    if (nrow(dtm_new_data) == 1) {
      result <- result / sum(result)
    } else {
      result <- result / Matrix::rowSums(result)
    }

    result <- result %*% t(object$lambda[, vocab_original])
    result <- as.matrix(result)
    
    repl <- is.na(result)
    
    bad_docs <- which(rowSums(repl) > 0)
    
    rownames(result) <- rownames(dtm_new_data)
    colnames(result) <- rownames(object$beta)
    
    # how do you want to handle empty documents?
    if (no_common_tokens[1] %in% c("default", "zero")) {
      if (length(bad_docs) > 0) {
        result[repl] <- 0
        if (no_common_tokens[1] == "default") {
          for (bad in bad_docs) {
            warning(
              "Document ", bad, " has no tokens in common with the model. ",
              "Setting predictions to 0 for all documents. To change this behavior ",
              "or silence this warning, change the value of 'no_common_tokens' in ",
              "the call to predict.tidylda."
            )
          }
        } else {
          for (bad in bad_docs) {
            message(
              "Document ", bad, " has no tokens in common with the model. ",
              "Setting predictions to 0 for all documents."
            )
          }
        } 
      } 
    } else { # means no_common_tokens == "uniform"
      result[bad_docs, ] <- 1 / ncol(object$theta)
    }
  } else { # gibbs method
    # format inputs

    # get initial distribution with recursive call to "dot" method
    theta_initial <- predict.tidylda(
      object = object, 
      new_data = new_data, 
      method = "dot", 
      no_common_tokens = "uniform"
    )

    # make sure priors are formatted correctly
    eta <- format_eta(object$eta, k = nrow(object$beta), Nv = ncol(dtm_new_data))

    alpha <- format_alpha(object$alpha, k = nrow(object$beta))

    # get initial counts
    counts <- initialize_topic_counts(
      dtm = dtm_new_data,
      k = nrow(object$beta),
      alpha = alpha$alpha,
      eta = eta$eta,
      beta_initial = object$beta,
      theta_initial = theta_initial,
      freeze_topics = TRUE,
      threads = threads,
    )

    # pass inputs to C++ function for prediciton
    lda <- fit_lda_c(
      Docs = counts$Docs,
      Zd_in = counts$Zd,
      Cd_in = counts$Cd,
      Cv_in = counts$Cv,
      Ck_in = counts$Ck,
      alpha_in = alpha$alpha,
      eta_in = eta$eta,
      iterations = iterations,
      burnin = burnin,
      optimize_alpha = FALSE,
      calc_likelihood = FALSE,
      Beta_in = object$beta, 
      freeze_topics = TRUE,
      threads = threads,
      verbose = verbose
    )
    

    # format posterior prediction
    result <- new_tidylda(
      lda = lda, 
      dtm = dtm_new_data,
      burnin = burnin, 
      is_prediction = TRUE, 
      threads
    )
  }

  # If type is "class" or "distribution", format further
  if (type[1] == "class") {
    
    result <- apply(result, 1, function(x) which.max(x)[1])
    
  } else if (type[1] == "distribution") {
    
    dir_par <- result * (Matrix::rowSums(dtm_new_data) + sum(object$alpha))
    
    dir_par <- t(dir_par)
    
    result <- generate_sample(
      dir_par = dir_par,
      matrix = "theta",
      times = times
    )
    
  }
  
  # return result
  result
}
