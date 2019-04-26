#pragma once

//Ospre
#ifdef FLUXLOSPRE
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r<1.e10) return 1.5*(r*r+r)/(r*r+r+1.0);
  else return 1.5;
}
#endif

//van albda 1
#ifdef FLUXLVANALBDA1
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r<1.e10) return (r*r+r)/(r*r+1.0);
  else return 1.0;
}
#endif

//van albda 2
#ifdef FLUXLVANALBDA2
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r<1.e10) return (2.0*r)/(r*r+1.0);
  else return 0.0;
}
#endif

//superbee
#ifdef FLUXLSUPERBEE
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (2.0*r < 1.0) return 2.0*r;
  else if (r < 2.0) return r;
  else return 2.0;
}
#endif

//Koren (1993)
#ifdef FLUXLKOREN
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r<0.4) return 2.0*r;
  else if (r<=1.0) return (2.0+r)/3.0;
  else return 2.0;
}
#endif

//Osher
#ifdef FLUXLOSHER
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r < 1.5) return r;
  else return 1.5;
}
#endif

//Generalised minmod
#ifdef FLUXLGMINMOD
#define FLUXL
  #ifndef THETA
  #define THETA 1.5
  #endif
inline const double phi(const double &r)
{
  if ((THETA*r < 0.0) || (0.5*(1.0+r) < 0.0)) return 0.0;
  else if ((THETA*r < 0.5*(1.0+r)) && (THETA*r < THETA)) return THETA*r;
  else if ((0.5*(1.0+r) < THETA*r) && (0.5*(1.0+r) < THETA)) return 0.5*(1.0+r);
  else return THETA;
}
#endif

//CHARM
#ifdef FLUXLCHARM
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r>0.0) return r*(3.0*r+1.0)/((r+1.0)*(r+1.0));
  else return 3.0;
}
#endif

//monotonized central
#ifdef FLUXLMONOCENTRAL
#define FLUXL
inline const double phi(const double &r)
{
  if (r < 0.0) return 0.0;
  else if (r < one_third) return 2.0*r;
  else if (r <= 1.0) return 0.5*(1.0+r);
  else return 2.0;
}
#endif

//minmod
#ifdef FLUXLMINMOD
#define FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r<1.0) return r;
  else return 1.0;
}
#endif

//minmod if no limiter was chosen
#ifndef FLUXL
inline const double phi(const double &r)
{
  if (r<0.0) return 0.0;
  else if (r<1.0) return r;
  else return 1.0;
}
#endif