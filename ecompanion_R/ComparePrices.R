#==========================================================#
#====================== May 25, 2025 ======================#
#========== Simulations - Compare Optimal Prices ==========#
#==========================================================#

library(pracma)


# repetition.RANDOM.SEED = 9
repetition.RANDOM.SEED = as.numeric(Sys.getenv("SGE_TASK_ID"))



start_time <- Sys.time()

#--- precision
epsilon = 0.01
PriceUB = 100

#------------------------#
#--- Define variables ---#
#------------------------#

N.prods.range = c(5, 10, 20)
K.attributes.range = c(1, 3, 5)

parameter.index = 1
instance.index = NA #the first row has an index starting from 0
how.many.instances = 100

N.index = ceiling(repetition.RANDOM.SEED / 3)
K.index = repetition.RANDOM.SEED - (N.index - 1) * 3

N.prods = N.prods.range[N.index]
K.attributes = K.attributes.range[K.index]
L.levels = 3

a.vec = rep(NA, N.prods)
b.vec = rep(NA, N.prods)
c.vec = rep(NA, N.prods)
gamma.vec = rep(NA, (K.attributes+1))
eta.vec = rep(1, (K.attributes+1))
theta.matrix = matrix(NA, nrow = N.prods, ncol = K.attributes)

#--------------------------------------#
#--- Define variables for MNL model ---#
#--------------------------------------#

beta.MNL.vec = rep(NA, N.prods)
c.MNL.vec = rep(NA, N.prods)
theta.MNL.matrix = matrix(NA, nrow = N.prods, ncol = K.attributes)

#-----------------------------------#
#--- some functions for CC model ---#
#-----------------------------------#

profit <- function(p.vec_)
{
  pmin = min(p.vec_)
  M.vec = M.tilde.vec - gamma0 * (p.vec_ - pmin)^eta0
  profit = ((p.vec_ - c.vec) %*% exp(a.vec - b * p.vec_ + M.vec)) / (1 + sum(exp(a.vec - b * p.vec_ + M.vec))) 
  return(profit)
}

utility <- function(p.vec_) # this actually means "surplus" --- all occurences of "utility" refers to surplus!
{
  pmin = min(p.vec_)
  M.vec = M.tilde.vec - gamma0 * (p.vec_ - pmin)^eta0
  returned = log(  1 + sum(exp(a.vec - b * p.vec_ + M.vec))  )
  return(returned)
}

g.function <- function(p_, j, x_, V_)
{
  returned = (p_ - c.vec[j] - V_) * (eta0 * gamma0 * (p_ - x_)^(eta0 - 1) + b)
  return(returned)
}

find.root <- function(g.function_, j, x_, V_, p.bar_ = p.bar)
{
  effective.range = seq(0, p.bar_, epsilon)
  g.values = g.function_(effective.range, j, x_, V_)
  closest.index = which.min(abs(g.values - 1))
  returned = effective.range[closest.index]
  if(returned < x_)
  {
    returned = x_
  }
  return(returned)
}

#------------------------------------#
#--- some functions for MNL model ---#
#------------------------------------#

MNL_profit <- function(markup_)
{
  p.MNL.vec = markup_ + c.MNL.vec
  profit = (markup_ * sum(exp(beta0.MNL + betaP.MNL * p.MNL.vec + M.MNL.vec))) / (1 + sum(exp(beta0.MNL + betaP.MNL * p.MNL.vec + M.MNL.vec))) 
  return(profit)
}

MNL_find_max_profit <- function(epsilon_ = epsilon)
{
  markup.list = seq(epsilon_, PriceUB, epsilon_)
  profits = sapply(markup.list, MNL_profit)
  
  max_profit = max(profits)
  max_index = which.max(profits)
  max_markup = markup.list[max_index]
  
  return(list(max_profit = max_profit,
              max_markup = max_markup))
}

#------------------------------#
#--- Read data for CC model ---#
#------------------------------#

if(N.prods == 5)
{
  N.prods_name = "05"
}
if(N.prods == 10)
{
  N.prods_name = "10"
}
if(N.prods == 20)
{
  N.prods_name = "20"
}

file_name = paste0("Instances/CC_parameters_", N.prods_name, "prod_", K.attributes, "attr.csv")
read.data = read.csv(file_name)
# View(read.data)
read.matrix = as.matrix(read.data)
# View(read.matrix)
storage.mode(read.matrix) = "numeric"

#-------------------------------#
#--- Read data for MNL model ---#
#-------------------------------#

file_name.MNL = paste0("Instances/MNL_parameters_", N.prods_name, "prod_", K.attributes, "attr.csv")
read.data.MNL = read.csv(file_name.MNL)
read.matrix.MNL = as.matrix(read.data.MNL)
storage.mode(read.matrix.MNL) = "numeric"

#==================================#
#========== Solve prices ==========#
#==================================#

CC_utility = rep(NA, how.many.instances)
CC_max_profit = rep(NA, how.many.instances)
CC_optimal_prices = matrix(NA, nrow = how.many.instances, ncol = N.prods)

MNL_max_profit = rep(NA, how.many.instances)
CC_profit_under_MNL_prices = rep(NA, how.many.instances)
CC_utility_under_MNL_prices = rep(NA, how.many.instances)
MNL_optimal_prices = matrix(NA, nrow = how.many.instances, ncol = N.prods)

for(instance.index in 1:how.many.instances)
{
  #========================================#
  #=============== CC model ===============#
  #========================================#
  
  #---------------------------------------#
  #--- Pass values to random variables ---#
  #---------------------------------------#
  
  c.start.index = 4 # 4 columns before; remember to add 1!
  c.vec = read.matrix[instance.index, (c.start.index+1):(c.start.index+N.prods)]
  
  theta.start.index = 4 + N.prods + N.prods # 4 columns, then N.prods columns for c and N.prods columns for p; remember to add 1!
  for(K.temp in 1:K.attributes)
  {
    theta.matrix[, K.temp] = read.matrix[instance.index, (theta.start.index+1+(K.temp-1)*N.prods):(theta.start.index+K.temp*N.prods)]
  }
  
  a.start.index = theta.start.index + N.prods * K.attributes # N.prods * K.attributes columns for theta; remember to add 1!
  a.vec = read.matrix[instance.index, (a.start.index+1):(a.start.index+N.prods)]
  
  b.start.index = a.start.index + N.prods # N.prods columns for a; remember to add 1!
  b.vec = read.matrix[instance.index, (b.start.index+1):(b.start.index+N.prods)]
  b = b.vec[1]
  
  gamma.start.index = b.start.index + N.prods # N.prods columns for b; remember to add 1!
  gamma0 = read.matrix[instance.index, (gamma.start.index+1)]
  gamma.vec = read.matrix[instance.index, (gamma.start.index+2):(gamma.start.index+1+K.attributes)]
  
  eta.start.index = gamma.start.index + 1 + K.attributes # 1 + K.attributes columns for gamma (including gamma0); remember to add 1!
  eta0 = read.matrix[instance.index, (eta.start.index+1)]
  eta.vec = read.matrix[instance.index, (eta.start.index+2):(eta.start.index+1+K.attributes)]
  
  #-------------------------------------------#
  #--- Prepare some intermediate variables ---#
  #-------------------------------------------#
  
  min.removed.matrix = matrix(NA, nrow = N.prods, ncol = K.attributes)
  theta.min.vec = apply(theta.matrix, 2, min)
  for(K.temp in 1:K.attributes)
  {
    min.removed.matrix[, K.temp] = (theta.matrix[,K.temp] - theta.min.vec[K.temp])^eta.vec[K.temp]
  }
  
  M.tilde.vec = rowSums(min.removed.matrix %*% diag(gamma.vec, nrow = K.attributes, ncol = K.attributes))
  
  psi.vec = exp(a.vec + M.tilde.vec)
  
  v.bar = lambertWp(sum(exp(a.vec - b * c.vec - 1 + M.tilde.vec)))
  p.bar.comp1 = max(c.vec) + v.bar + (1 / b)
  p.bar.comp2 = (1 + sum(psi.vec) + b * sum(c.vec * psi.vec)) / (b * sum(psi.vec))
  p.bar.comp3 = 2 * log(sum(psi.vec)) / b
  p.bar = max(p.bar.comp1, p.bar.comp2, p.bar.comp3)
  
  Delta.err = 2 * sum(psi.vec)
  
  price.range = seq(0, p.bar, by = epsilon)
  profit.range = seq(0, v.bar, by = epsilon)
  max.profit.GridSearch = -Inf
  optimal.prices.GridSearch = c()
  for(x.temp in (price.range[-length(price.range)]))
  {
    for(V.temp in profit.range)
    {
      p.tilde.vec = c()
      for(N.temp in 1:N.prods)
      {
        p.tilde.vec = c(p.tilde.vec, find.root(g.function, N.temp, x.temp, V.temp))
      }
      current_profit = profit(p.tilde.vec)
      # Update maximum profit and corresponding prices if the current profit is higher
      if(current_profit > max.profit.GridSearch)
      {
        max.profit.GridSearch = current_profit
        optimal.prices.GridSearch = p.tilde.vec
      }
    }
  }
  
  CC_max_profit[instance.index] = profit(optimal.prices.GridSearch)
  CC_optimal_prices[instance.index, ] = optimal.prices.GridSearch
  CC_utility[instance.index] = utility(optimal.prices.GridSearch)
  
  #=========================================#
  #=============== MNL model ===============#
  #=========================================#
  
  #---------------------------------------#
  #--- Pass values to random variables ---#
  #---------------------------------------#
  
  c.start.index = 4 # 4 columns before; remember to add 1!
  c.MNL.vec = read.matrix.MNL[instance.index, (c.start.index+1):(c.start.index+N.prods)]
  
  theta.start.index = 4 + N.prods + N.prods # 4 columns, then N.prods columns for c and N.prods columns for p; remember to add 1!
  for(K.temp in 1:K.attributes)
  {
    theta.MNL.matrix[, K.temp] = read.matrix.MNL[instance.index, (theta.start.index+1+(K.temp-1)*N.prods):(theta.start.index+K.temp*N.prods)]
  }
  
  beta.start.index = theta.start.index + N.prods * K.attributes # N.prods * K.attributes columns for theta; remember to add 1!
  beta0.MNL = read.matrix.MNL[instance.index, (beta.start.index+1)]
  betaP.MNL = read.matrix.MNL[instance.index, (beta.start.index+2)]
  beta.MNL.vec = read.matrix.MNL[instance.index, (beta.start.index+3):(beta.start.index+K.attributes+2)]
  
  #-------------------------------------------#
  #--- Prepare some intermediate variables ---#
  #-------------------------------------------#
  
  M.MNL.vec = rowSums(theta.MNL.matrix %*% diag(beta.MNL.vec, nrow = K.attributes, ncol = K.attributes))
  
  MNL.solved = MNL_find_max_profit()
  
  # MNL_max_profit[instance.index] = MNL.solved$max_profit
  MNL.prices = MNL.solved$max_markup + c.MNL.vec
  MNL_optimal_prices[instance.index, ] = MNL.prices
  CC_profit_under_MNL_prices[instance.index] = profit(MNL.prices)
  CC_utility_under_MNL_prices[instance.index] = utility(MNL.prices)
}

CC_output_matrix = cbind(seq(1, how.many.instances), 
                         rep(N.prods, how.many.instances),
                         rep(K.attributes, how.many.instances),
                         rep(L.levels, how.many.instances),
                         CC_max_profit,
                         CC_optimal_prices,
                         CC_utility)
colnames(CC_output_matrix) = c(c("Instance", "N", "K", "L", "Exp Profit"), paste0("p_", 1:N.prods), "surplus")

CC_output_filename = paste0("Results/CC_OptPrices_", N.prods_name, "prod_", K.attributes, "attr.csv")
write.csv(CC_output_matrix, file = CC_output_filename, row.names = FALSE)



MNL_output_matrix = cbind(seq(1, how.many.instances), 
                          rep(N.prods, how.many.instances),
                          rep(K.attributes, how.many.instances),
                          rep(L.levels, how.many.instances),
                          CC_profit_under_MNL_prices,
                          MNL_optimal_prices,
                          CC_utility_under_MNL_prices)
colnames(MNL_output_matrix) = c(c("Instance", "N", "K", "L", "Exp Profit"), paste0("p_", 1:N.prods), "surplus")

MNL_output_filename = paste0("Results/MNL_OptPrices_", N.prods_name, "prod_", K.attributes, "attr.csv")
write.csv(MNL_output_matrix, file = MNL_output_filename, row.names = FALSE)



# summary(CC_profit_under_MNL_prices / CC_max_profit)
# summary(CC_utility_under_MNL_prices / CC_utility)





end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)
