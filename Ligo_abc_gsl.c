// Compile with: gcc-8 -O2 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF Ligo_abc_gsl.c -o Ligo_abc_gsl.exe -lgsl -lgslcblas -fopenmp
// Run with: ./Ligo_abc_gsl.exe
// Test running time: time ./Ligo_abc_gsl.exe

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <sys/time.h>
#include <omp.h>

#define CSV 1

// PROGRAM VARIABLES
// Variables for parallelization with OpenMP
// Info on OpenMP: https://bisqwit.iki.fi/story/howto/openmp/#Syntax
int OPENMP = 1;
int OMP_THREAD_NUM = 8;

// Make the Lag test (can make the computation very long!)
int LAG_TEST = 0;
// Ratio of lag function computed with respect to the total number of points in every chain
double LAG_TEST_STOP_RATIO = 0.001;


// NUMERICAL VARIABLES
// Chain evaluation dimension (every step found from proposal distribution is multiplied by this quantity)
// This is also used to define the covariance matrix for the Gaussian proposal
double SPEED[] = {1.,1e-15,1.};
// Mean value for Gaussian proposal (PROPOSAL=1)
double MEAN_P1[] = {0.,0.,0.};
// Number of points in every chain
int N = 1e6;

// PHYSICAL VARIABLES
// Choose to use the constraint on Ct2 in the prior
int CT2_CONSTR = 1;
// Choose the proposal distribution
int PROPOSAL = 1;







typedef struct chain {
  gsl_vector * mean;
  gsl_matrix * covariance;
} chain_res;


inline double cT2 (gsl_vector * pos) {

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  return (2.0-2.0*a)/(2.0-2.0*a+b);

}

inline double cS2 (gsl_vector * pos) {

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  return (b-c)*(2.0*a-2.0)/(2.0*a-b-2.0)/(4.0-4.0*a-b-3.0*c);

}


inline double rho(gsl_vector * pos){

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  // Value of the sound speed squared
  double cs2=cS2(pos);
  // Value of the tensor speed squared
  double ct2=cT2(pos);
  // Value of the argument of the Gaussian in the Likelihood
  double delta=fabs(sqrt(ct2)-1.0);

  // This is the experimental sigma on cT2
  double sigma = 4.5e-16;

  double result = 0.;

  // With these ifs I'm introducing the constraints on cT2, cS2 and on the parameters a and c
  // To eliminate these constraints, put CT2_CONSTR=0 at the beginning of the file
  //
  // OLD constraint: CT2_CONSTR==1 && a>=-1. && a<=1. && c>=-4.0/3.0 && c<=0. && cT2>=0.0 && cT2<= 1.0 && cS2>=0.0 && cS2<=1.0
  //
  // NEW (correct) constraint
  if (CT2_CONSTR==1 && a>=-1. && a<=1. && c>=0. && ct2>=0.0 && ct2<= 1.0 && cs2<=0.0){
    result=exp(-0.5 * pow(delta/sigma,2.0) );
  }
  else if(CT2_CONSTR==1){
    result=exp(-1e90);
  }
  else{
    result=exp(-0.5 * pow(delta/sigma,2.0) );
  }

  return result;
}

chain_res chain_analysis(double (* chain)[3]){

  chain_res result;

  gsl_vector * mean = gsl_vector_alloc(3);

  int i,k,s;
  for(k=0;k<3;k++){
    double mean_temp = 0.;
    for(i=0;i<N;i++){
      mean_temp = mean_temp + chain[i][k];
    }
    gsl_vector_set(mean, k, mean_temp);
  }
  gsl_vector_scale(mean,1./N);

  gsl_matrix * cm = gsl_matrix_alloc(3,3);

  for(k=0;k<3;k++){
    double mean_row = gsl_vector_get(mean,k);
    for(s=0;s<3;s++){
      double mean_column = gsl_vector_get(mean,s);
      double cm_temp = 0.;
      for(i=0;i<N;i++){
        cm_temp = cm_temp + (chain[i][k]-mean_row)*(chain[i][s]-mean_column);
      }
      gsl_matrix_set(cm, k, s, cm_temp);
    }
  }
  gsl_matrix_scale(cm,1./(N-1.));

  result.mean = gsl_vector_alloc(3);
  gsl_vector_memcpy(result.mean,mean);
  result.covariance = gsl_matrix_alloc(3,3);
  gsl_matrix_memcpy(result.covariance,cm);

  gsl_matrix_free(cm);
  gsl_vector_free(mean);

  return result;

}

inline double Gaussian_proposal(double mean, double sigma, double u1, double u2) {

  // Box Muller transform from flat distribution to Guassian distribution
  return mean + sigma * sqrt(-2.*log(u1)) * cos(2.*M_PI*u2);

}

void C_Delta(double (* chain)[3], chain_res results, double f_stop){

  int id_thread = omp_get_thread_num();

  double mean_X = gsl_vector_get(results.mean,0);
  double mean_Y = gsl_vector_get(results.mean,1);
  double mean_Z = gsl_vector_get(results.mean,2);

  double cm_XX = gsl_matrix_get(results.covariance,0,0);
  double cm_YY = gsl_matrix_get(results.covariance,1,1);
  double cm_ZZ = gsl_matrix_get(results.covariance,2,2);

  int stop = N*f_stop;

  double * C_Delta_X; double * C_Delta_Y; double * C_Delta_Z;
  C_Delta_X = (double *) malloc(sizeof(double) * stop);
  C_Delta_Y = (double *) malloc(sizeof(double) * stop);
  C_Delta_Z = (double *) malloc(sizeof(double) * stop);

  int i,j;
  for(i=0;i<stop;i++){
    for(j=0;j<N-i;j++){
      C_Delta_X[i]=C_Delta_X[i]+(chain[j][0]-mean_X)*(chain[j+i][0]-mean_X);
      C_Delta_Y[i]=C_Delta_Y[i]+(chain[j][1]-mean_Y)*(chain[j+i][1]-mean_Y);
      C_Delta_Z[i]=C_Delta_Z[i]+(chain[j][2]-mean_Z)*(chain[j+i][2]-mean_Z);
    }
    C_Delta_X[i]=C_Delta_X[i]/(N-i)/cm_XX;
    C_Delta_Y[i]=C_Delta_Y[i]/(N-i)/cm_YY;
    C_Delta_Z[i]=C_Delta_Z[i]/(N-i)/cm_ZZ;
  }

  FILE *fp;
  char filename[25];

  sprintf(filename,"Ligo_abc_DeltaC_t%d.csv",id_thread);
  fp = fopen (filename, "w+");

  for(i=0;i<stop;i++){
    fprintf(fp, "%d,%.8e,%.8e,%.8e\n", i, C_Delta_X[i], C_Delta_Y[i], C_Delta_Z[i]);
  }

  fclose(fp);

  free(C_Delta_X);
  free(C_Delta_Y);
  free(C_Delta_Z);

}

chain_res LHSampling(int proposal){

  int k;

  int id_thread=omp_get_thread_num();

  const gsl_rng_type * T;
  gsl_rng * r;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  unsigned long seed = time(NULL) + id_thread;

  gsl_rng_set(r, seed);

  gsl_vector * pos_ini = gsl_vector_alloc(3);
  for(k=0;k<3;k++){
    if(k==0){
      double random_number = 2. * gsl_rng_uniform (r) - 1.;
      gsl_vector_set(pos_ini,k,SPEED[k]*random_number);
    }
    else if (k==1){
      double random_number = gsl_rng_uniform (r);
      gsl_vector_set(pos_ini,k,SPEED[k]*random_number);
    }
    else {
      double random_number = gsl_rng_uniform (r);
      gsl_vector_set(pos_ini,k,SPEED[k]*random_number);
    }
  }

  // Here definitions only for the Gaussian proposal distribution
  gsl_vector * pos_mean_p1 = gsl_vector_alloc(3);
  for(k=0;k<3;k++){
    gsl_vector_set(pos_mean_p1, k, MEAN_P1[k]);
  }

  gsl_matrix * S = gsl_matrix_calloc(3,3);
  gsl_matrix * L = gsl_matrix_alloc(3,3);
  gsl_vector * pos_temp2 = gsl_vector_alloc(3);

  gsl_matrix_set(S,0,0,pow(SPEED[0],2.));
  gsl_matrix_set(S,1,1,pow(SPEED[1],2.));
  gsl_matrix_set(S,2,2,pow(SPEED[2],2.));
  gsl_matrix_memcpy(L,S);
  gsl_linalg_cholesky_decomp1(L);
  // End of Gaussian proposal definitions

  gsl_vector * pos = gsl_vector_alloc(3);
  gsl_vector_memcpy(pos,pos_ini);

  #if CSV==1
    FILE *fp;
    char filename[25];
    sprintf(filename,"Ligo_abc_%d.txt",id_thread);
    fp = fopen (filename, "w+");
  #endif

  double (* chain)[3] = malloc(sizeof(* chain) * N);

  int i = 0; int j = 0; int rc = 1;

  while (i<N) {

    double f0 = rho(pos);

    gsl_vector * pos_temp = gsl_vector_calloc(3);

    // NOTE: Introducing the proposal distributions here

    if(proposal==0){

      for(k=0;k<3;k++){

        double random_number = 2. * gsl_rng_uniform (r) - 1.;
        gsl_vector_set(pos_temp, k, SPEED[k]*random_number);

      }

      gsl_vector_add(pos_temp,pos);

    }

    else if(proposal=!0){

      for(k=0;k<3;k++){

        gsl_vector_set(pos_temp2, k, Gaussian_proposal(0., 1., gsl_rng_uniform (r), gsl_rng_uniform (r)));

      }

      gsl_blas_dgemv (CblasNoTrans,
                  1., L, pos_temp2,
                  0., pos_temp);
      gsl_vector_add(pos_temp,pos_mean_p1);
      gsl_vector_add(pos_temp,pos);

    }

    double f1 = rho(pos_temp);

    // NOTE: here we also print data in .csv file (if CSV==1) in Getdist ready format
    // Getdist output with (no commas): repetitions count log(rho) {parameters} {(optional) other outputs - not considered by Getdist}

    if (f1>f0) {

      gsl_vector_memcpy(pos,pos_temp);
      j++;

      #if CSV==1
      fprintf(fp, "%d %.8e %.8e %.8e %.8e\n", rc, -log(rho(pos)), gsl_vector_get(pos,0), gsl_vector_get(pos,1), gsl_vector_get(pos,2));
      #endif

      rc = 1;

    }
    else {

      double ra = gsl_rng_uniform (r);

      if (ra<f1/f0) {
        gsl_vector_memcpy(pos,pos_temp);
        j++;

        #if CSV==1
        fprintf(fp, "%d %.8e %.8e %.8e %.8e\n", rc, -log(rho(pos)), gsl_vector_get(pos,0), gsl_vector_get(pos,1), gsl_vector_get(pos,2));
        #endif

        rc = 1;
      }
      else{
        rc++;
      }

    }

    for(k=0;k<3;k++){
      chain[i][k]=gsl_vector_get(pos,k);
    }

    gsl_vector_free(pos_temp);
    i++;

  }

  chain_res result;

  result = chain_analysis(chain);

  printf("\n Acceptance ratio: %f \n\n", (double) j/i);
  printf(" Results: mean value pm sqrt(covariant_ii)\n");
  printf(" \ta: %e pm %e\n", gsl_vector_get(result.mean,0), sqrt(fabs(gsl_matrix_get(result.covariance,0,0))));
  printf(" \tb: %e pm %e\n", gsl_vector_get(result.mean,1), sqrt(fabs(gsl_matrix_get(result.covariance,1,1))));
  printf(" \tc: %e pm %e\n", gsl_vector_get(result.mean,2), sqrt(fabs(gsl_matrix_get(result.covariance,2,2))));

  printf("\n Covariance coefficients (Pearson coefficients : covariant_ij / sigma_i sigma_j): \n");
  printf(" \tAB: %e \n", gsl_matrix_get(result.covariance,0,1)/sqrt(fabs(gsl_matrix_get(result.covariance,0,0)))/sqrt(fabs(gsl_matrix_get(result.covariance,1,1))));
  printf(" \tAC: %e \n", gsl_matrix_get(result.covariance,0,2)/sqrt(fabs(gsl_matrix_get(result.covariance,0,0)))/sqrt(fabs(gsl_matrix_get(result.covariance,2,2))));
  printf(" \tBC: %e \n\n", gsl_matrix_get(result.covariance,1,2)/sqrt(fabs(gsl_matrix_get(result.covariance,1,1)))/sqrt(fabs(gsl_matrix_get(result.covariance,2,2))));

  // Check of correlation (with lag)
  if(LAG_TEST==1){

    printf(" Computing Lag function..\n");
    C_Delta(chain,result,LAG_TEST_STOP_RATIO);

  }


  // Free all the variables
  free(chain);

  gsl_vector_free(pos_ini);
  gsl_vector_free(pos_mean_p1);
  gsl_vector_free(pos);
  gsl_rng_free(r);

  gsl_vector_free(pos_temp2);
  gsl_matrix_free(S);
  gsl_matrix_free(L);

  #if CSV==1
    fclose(fp);
  #endif

  return result;

}

inline double R(double sigma2_chain, double sigma2_mean){

  int noc = OMP_THREAD_NUM;

  return sqrt(((N-1.)/N*sigma2_chain + (noc+1.)/N/noc*sigma2_mean)/sigma2_chain);

}

void GR_Test(chain_res * chain_results) {

  int i, k;
  int noc = OMP_THREAD_NUM;

  gsl_vector * mean = gsl_vector_alloc(3);
  gsl_vector * sigma2_chain = gsl_vector_alloc(3);
  gsl_vector * sigma2_mean = gsl_vector_alloc(3);

  for(k=0;k<3;k++){
    double mean_temp = 0.;
    double sigma2_temp = 0.;
    for(i=0;i<noc;i++){
      mean_temp = mean_temp + gsl_vector_get(chain_results[i].mean,k)/noc;
      sigma2_temp = sigma2_temp + fabs(gsl_matrix_get(chain_results[i].covariance,k,k))/noc;
    }
    gsl_vector_set(mean, k, mean_temp);
    gsl_vector_set(sigma2_chain, k, sigma2_temp);
  }

  for(k=0;k<3;k++){
    double mean_temp = 0.;
    double sigma2_temp = 0.;
    for(i=0;i<noc;i++){
      sigma2_temp = sigma2_temp + pow(gsl_vector_get(mean,k)-gsl_vector_get(chain_results[i].mean,k),2.)*N/(noc-1.);
    }
    gsl_vector_set(sigma2_mean, k, sigma2_temp);
  }

  printf(" Gelman-Rubin test.\n");
  printf(" \t- R_a : %f\n", R(gsl_vector_get(sigma2_chain,0), gsl_vector_get(sigma2_mean,0)));
  printf(" \t- R_b : %f\n", R(gsl_vector_get(sigma2_chain,1), gsl_vector_get(sigma2_mean,1)));
  printf(" \t- R_c : %f\n", R(gsl_vector_get(sigma2_chain,2), gsl_vector_get(sigma2_mean,2)));
  printf(" Converge if (approximately) R < 1.2. For better results R < 1.01.\n");
  printf(" -------------------------------------------------- \n");

  gsl_vector_free(mean);
  gsl_vector_free(sigma2_chain);
  gsl_vector_free(sigma2_mean);

}

void init_msg(int proposal){

  printf(" Starting the MCMC computation with Metropolis-Hastings algorithm.\n");
  printf(" Properties of the chains:\n");
  if(proposal==0){
    printf(" \t- Proposal distribution: flat.\n");
  }
  if(proposal!=0){
    printf(" \t- Proposal distribution: Gaussian.\n");
  }
  if(OPENMP==0){
    printf(" \t- The chain will be %d points long.\n", N);
  }
  if(OPENMP==1){
    printf(" \t- Each of the %d chains will be %d points long.\n", omp_get_max_threads(), N);
  }
  printf(" -------------------------------------------------- \n");

}

int main(){

  int proposal=PROPOSAL;

  printf(" -------------------------------------------------- \n");

  if(OPENMP==1){

    chain_res * chain_results;
    chain_results = (chain_res *) malloc(sizeof(chain_res) * OMP_THREAD_NUM);

    printf(" Info on OpenMP on this computer. \n");
    printf(" \t- Number of processors available = %d\n", omp_get_num_procs ( ) );
    printf(" \t- Number of threads              = %d\n", omp_get_max_threads ( ) );
    printf(" \t- Number of working threads      = %d\n", OMP_THREAD_NUM );
    printf(" Specify the number of working threads in .c file.\n");
    printf(" -------------------------------------------------- \n");

    if(OMP_THREAD_NUM>omp_get_max_threads()){
      printf(" Error. Can't set number of threads (OMP_THREAD_NUM = %d) bigger than the available number of threads (%d). \n",OMP_THREAD_NUM, omp_get_max_threads());
      printf(" The program will be terminated. \n");
      exit(0);
    }


    init_msg(proposal);

    #pragma omp parallel num_threads(OMP_THREAD_NUM)
    {
      int current_thread = omp_get_thread_num();
      printf(" Thread %d working..\n", current_thread);
      chain_results[current_thread]=LHSampling(proposal);
      printf(" Thread %d has finished the evaluation of the chain. \n", current_thread);
      printf(" -------------------------------------------------- \n");
    }

    GR_Test(chain_results);

    free(chain_results);

  }
  else{

    init_msg(proposal);

    LHSampling(proposal);
    printf(" -------------------------------------------------- \n");

  }

  return 0;

}
