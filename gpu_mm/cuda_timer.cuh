#include <cuda_runtime.h>


class CudaTimer
{
private:
  cudaEvent_t _begEvent;
  cudaEvent_t _endEvent;

public:
  CudaTimer();
  ~CudaTimer();
  void start();
  void stop();
  float value();
};
