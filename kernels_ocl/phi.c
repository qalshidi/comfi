__kernel void fluxl(
          __global const double * vec,
          __global double * result,
          unsigned int size)
{
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
  {
    if (vec[i] < 0.0) { result[i] = 0.0; }
    else if (!isnan(vec[i])){ result[i] = 1.5*(vec[i]*vec[i]+vec[i])/(vec[i]*vec[i]+vec[i]+1.0); }
    else { result[i] = 1.5; }

    if(isnan(result[i])) { result[i] = 1.5; }
  }
};
