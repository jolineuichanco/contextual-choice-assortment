#=========================================#
#===== Read in all the files printed =====#
#=========================================#

library(tidyverse)

list_of_files_CC = list.files(path = "./Results/", recursive = TRUE, pattern = "^CC_.*\\.csv$", full.names = TRUE)
datalist_CC = lapply(list_of_files_CC, read.csv)

list_of_files_MNL = list.files(path = "./Results/", recursive = TRUE, pattern = "^MNL_.*\\.csv$", full.names = TRUE)
datalist_MNL = lapply(list_of_files_MNL, read.csv)

output_table = matrix(NA, nrow = 9, ncol = 11)
for(file.temp in 1:9)
{
  this.data_CC = datalist_CC[[file.temp]]
  this.matrix_CC = as.matrix(this.data_CC)
  storage.mode(this.matrix_CC) = "numeric"
  
  this.data_MNL = datalist_MNL[[file.temp]]
  this.matrix_MNL = as.matrix(this.data_MNL)
  storage.mode(this.matrix_MNL) = "numeric"
  
  cols_CC = ncol(this.matrix_CC)
  cols_MNL = ncol(this.matrix_MNL)
  
  profit_CC = this.matrix_CC[, 5]
  profit_MNL = this.matrix_MNL[, 5]
  loss = (profit_CC - profit_MNL) / profit_CC
  
  surplus_CC = this.matrix_CC[, cols_CC]
  surplus_MNL = this.matrix_MNL[, cols_MNL]
  
  prices_CC = this.matrix_CC[, c(6:cols_CC)]
  prices_MNL = this.matrix_MNL[, c(6:cols_MNL)]
  
  output_table[file.temp, 1] = mean(prices_CC)
  output_table[file.temp, 2] = min(prices_CC)
  output_table[file.temp, 3] = max(prices_CC)
  
  output_table[file.temp, 6] = mean(prices_MNL)
  output_table[file.temp, 7] = min(prices_MNL)
  output_table[file.temp, 8] = max(prices_MNL)
  
  output_table[file.temp, 4] = mean(surplus_CC)
  output_table[file.temp, 9] = mean(surplus_MNL)
  
  output_table[file.temp, 5] = mean(profit_CC)
  output_table[file.temp, 10] = mean(profit_MNL)
  output_table[file.temp, 11] = mean(loss) * 100
  
}

output_table = round(output_table, digits = 2)

write.csv(output_table, file = "summary_table.csv", row.names = FALSE)
