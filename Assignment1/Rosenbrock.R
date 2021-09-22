library(GA)
library(rgl)

f <- function(x, y)
{
z <- (1-x)^2+exp(1)*(y-x^2)^2
z
}

myMonitor <- function(obj){
  persp(x,y,z) -> res
  title(paste("iteration =", obj@iter), font.main = 1)
  points(trans3d(obj@population[,-2], obj@population[,-1], obj@fitness, pmat = res), col="red")
  Sys.sleep(0.2)
}

x <- seq(-1 ,1, length = 20)
y <- seq(-1, 1, length = 20)
z <- outer(x,y,f)

persp(x,y,z) -> res

GA <- ga(type = "real-valued", fitness = f, y, lower = c(-1,-1), upper = c(1,1), maxiter=500, crossover = gareal_blxCrossover, monitor = myMonitor)
a <- GA@solution[1]
b <- GA@solution[2]
c <- f(a,b)
points(trans3d(a,b,c, pmat = res), col = "green")

GA1 <- ga(type = "real-valued", fitness = f, y, lower = c(-1,-1), upper = c(1,1), maxiter=300, crossover = gareal_waCrossover, pmutation = 0.2, popSize = 100)
a <- GA1@solution[1]
b <- GA1@solution[2]
c <- f(a,b)
points(trans3d(a,b,c, pmat = res), col = "green")

GA2 <- ga(type = "real-valued", fitness = f, y, lower = c(-1,-1), upper = c(1,1), maxiter=300, crossover = gareal_laCrossover, pmutation = 0.3)
a <- GA2@solution[1]
b <- GA2@solution[2]
c <- f(a,b)
points(trans3d(a,b,c, pmat = res), col = "green")
