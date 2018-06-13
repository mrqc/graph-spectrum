import numpy

def adjFun(x, y):
  ret = numpy.empty((x.shape[0], y.shape[1]))
  threshold = 2
  for xIndex in range(0, x.shape[0]):
    threshold = 1 + numpy.random.poisson()
    for yIndex in range(0, y.shape[1]):
      val = numpy.random.normal()
      print val
      if val >= threshold:
        ret[xIndex][yIndex] = 1
      else:
        ret[xIndex][yIndex] = 0
  return ret

