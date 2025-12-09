#===========================================================#
#====================== March 9, 2024 ======================#
#========== Simulations - Algorithm 2 Grid Search ==========#
#===========================================================#

library(pracma)
set.seed(12345)

epsilon.OPT = 0.001

N.prods = 10
a.vec = rep(10, N.prods)
b = 5
c.vec = seq(N.prods,1)
K.attributes = 5
gamma.vec = 1 / (2^(0:(K.attributes-1)))
gamma0 = 1
eta.vec = rep(0.5, K.attributes)
eta0 = 0.5

theta.matrix = matrix(runif(N.prods * K.attributes, min = 0, max = 1), nrow = N.prods, ncol = K.attributes, byrow = TRUE)
theta.matrix = theta.matrix %*% diag((2^(0:(K.attributes-1))), nrow = K.attributes, ncol = K.attributes)

min.removed.matrix = matrix(NA, nrow = N.prods, ncol = 0)
theta.min.vec = apply(theta.matrix, 2, min)
for(K.temp in 1:K.attributes)
{
  min.removed.matrix = cbind(min.removed.matrix, (theta.matrix[,K.temp] - theta.min.vec[K.temp])^eta.vec[K.temp])
}

M.tilde.vec = rowSums(min.removed.matrix %*% diag(gamma.vec, nrow = K.attributes, ncol = K.attributes))

psi.vec = exp(a.vec + M.tilde.vec)

v.bar = lambertWp(sum(exp(a.vec - b * c.vec - 1 + M.tilde.vec)))
p.bar.comp1 = max(c.vec) + v.bar + (1 / b)
p.bar.comp2 = (1 + sum(psi.vec) + b * sum(c.vec * psi.vec)) / (b * sum(psi.vec))
p.bar.comp3 = 2 * log(sum(psi.vec)) / b
p.bar = max(p.bar.comp1, p.bar.comp2, p.bar.comp3)

# p.vec = c(0.1, 0.2, 0.3)
# pmin = min(p.vec)
# 
# M.vec = M.tilde.vec - gamma0 * (p.vec - pmin)^eta0
# profit = ((p.vec - c.vec) %*% exp(a.vec - b * p.vec + M.vec)) / (1 + sum(exp(a.vec - b * p.vec + M.vec))) 

profit <- function(p.vec_)
{
  pmin = min(p.vec_)
  M.vec = M.tilde.vec - gamma0 * (p.vec_ - pmin)^eta0
  profit = ((p.vec_ - c.vec) %*% exp(a.vec - b * p.vec_ + M.vec)) / (1 + sum(exp(a.vec - b * p.vec_ + M.vec))) 
  return(profit)
}

Delta.err = 2 * sum(psi.vec)

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





# #=======================================================================#
# #========== 1. Find the true optimal using brute force search ==========#
# #=======================================================================#
# 
# # Define the range of prices
# price.range.OPT = seq(0, p.bar, by = epsilon.OPT)
# 
# # Initialize variables to store maximum profit and corresponding prices
# max_profit <- -Inf
# optimal_prices <- c(0, 0, 0)
# 
# # Iterate over all possible combinations of prices
# for (p1 in price.range.OPT)
# {
#   for (p2 in price.range.OPT)
#   {
#     for (p3 in price.range.OPT)
#     {
#       current_profit = profit(c(p1, p2, p3))
#       # Update maximum profit and corresponding prices if the current profit is higher
#       if(current_profit > max_profit)
#       {
#         max_profit = current_profit
#         optimal_prices = c(p1, p2, p3)
#       }
#     }
#   }
# }
# 
# max_profit
# optimal_prices






#===========================================================#
#========== 2. Find the optimal using Algorithm 2 ==========#
#===========================================================#

start.time <- Sys.time()

epsilon = 0.001

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken



epsilon = 0.002

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch


epsilon = 0.005

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch


epsilon = 0.01

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch


epsilon = 0.02

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch


epsilon = 0.05

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch

epsilon = 0.1

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch

epsilon = 0.2

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch


epsilon = 0.5

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch


epsilon = 1

price.range = seq(0, p.bar, by = epsilon)
profit.range = seq(0, v.bar, by = epsilon)
max.profit.GridSearch = -Inf
optimal.prices.GridSearch = c(0, 0, 0)
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

max.profit.GridSearch
optimal.prices.GridSearch
