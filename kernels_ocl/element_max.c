__kernel void element_max(
          __global const double * vec1,
          __global const double * vec2,
          __global double       * result,
          unsigned int size)
{
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
  {
    if (vec1[i]*vec1[i] > vec2[i]*vec2[i]) { result[i] = vec1[i]; }
    else { result[i] = vec2[i]; }
  }
};
