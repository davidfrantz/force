#!/usr/bin/env Rscript

# load libraries ####################################################
library(dplyr)


# input #############################################################
args   <- commandArgs(trailingOnly = TRUE)
n_args <- 3

if (length(args) != n_args) {
  c(
    "\nWrong input!\n",
    " 1: path_hist\n",
    " 2: file_output\n"
  ) %>%
  stop()
}


library(dplyr)

standard_error <- 0.01
minsize <- 175

pixel_counts <- data.frame(
  value = 0:2, 
  count = c(
    1868242749,
    496472240,
    502851998
    )
)

users_accuracy <- data.frame(
  value = 0:2, 
  UA = c(
    0.9,
    0.9,
    0.9
  )
) 

df <- users_accuracy %>% 
  inner_join(
    pixel_counts,
    by = "value"
  )

if (nrow(df) != nrow(pixel_counts)){
  "input tables do not match\n"
  stop()
}

df <- df %>% 
  mutate(proportion = count / sum(count)) %>%
  mutate(stdev = sqrt(UA*(1-UA))) %>%
  mutate(WS = proportion * stdev)

number <- (sum(df$WS)/standard_error)**2
number

df <- df %>% 
  mutate(equal = round(number / nrow(df))) %>%
  mutate(proportional = round(number * proportion)) %>%
  mutate(compromise = NA)

if (min(df$proportional) < minsize){
  
  rare <- df %>% 
    filter(proportional < minsize) %>% 
    mutate(compromise = minsize)
  
  n_rare <- sum(rare$compromise)
  
  big <- df %>% 
    filter(proportional >= minsize) %>% 
    mutate(compromise = (number-n_rare) * proportion)

  if (any(big$compromise < minsize)){
    c(
      "compromising the sample allocation failed...\n",
      "either decrease minsize or adjust input parameters to \n",
      "have more samples overall."
      )
  }

  df <- rbind(rare, big)

}

df %>%
    mutate(deviation = (compromise - proportional) / proportional * 100)

write.csv(df, file_output, row.names = FALSE, quote = FALSE)



