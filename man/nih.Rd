% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tidylda-package.R
\docType{data}
\name{nih}
\alias{nih}
\alias{nih_sample}
\alias{nih_sample_dtm}
\title{Abstracts and metadata from NIH research grants awarded in 2014}
\format{
For \code{nih_sample}, a \code{\link[tibble]{tibble}} of 100 randomly-sampled
grants' abstracts and metadata. For \code{nih_sample_dtm}, a
\code{\link[Matrix]{dgCMatrix-class}} representing the document term matrix
of abstracts from 100 randomly-sampled grants.
}
\source{
National Institutes of Health ExPORTER
\url{https://reporter.nih.gov/exporter}
}
\usage{
data("nih_sample")
}
\description{
This dataset holds information on research grants awarded by the National
Institutes of Health (NIH) in 2014. The data set was downloaded in
approximately January of 2015. It includes both 'projects' and 'abstracts' 
files.
}
