// mimetic_bayes program
// Alessandro Casalino, University of Trento
// For licence informations, see the Github repository license
//
// Compile with: gcc-8 -O2 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF Ligo_abc_gsl.c -o Ligo_abc_gsl.o -lgsl -lgslcblas -fopenmp
// Run with: ./Ligo_abc_gsl.o
// Test running time: time ./Ligo_abc_gsl.o

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <sys/time.h>
#include <omp.h>

// PROGRAM VARIABLES

// Variables for parallelization with OpenMP
// Info on OpenMP: https://bisqwit.iki.fi/story/howto/openmp/#Syntax
// Enable OpenMP
int OPENMP = 1;
// Number of threads used by OpenMP (can't be bigger than the number of threads available on your computer)
// If 0, the number of threads available on the computer is used
int OMP_THREAD_NUM = 0;

// Number of Chains (for higher efficiency, consider N_CHAIN as a multiple of OMP_THREAD_NUM)
int N_CHAINS = 8;

// Make the Lag test (can make the computation very long!)
int LAG_TEST = 0;
// Ratio of lag function computed with respect to the total number of points in every chain
// Smaller values make the computation faster
double LAG_TEST_STOP_RATIO = 0.001;

// Write (in the terminal) debug informations and informations that might be considered in order to improve results
int DEBUG = 1;

// Write .csv files
#define CSV 1
// Write covariant matrix in .csv file
int CSV_COV = 1;
// Write correlation matrix (Pearson coefficients) in .csv file
int CSV_CORR = 1;


// NUMERICAL VARIABLES
// Chain evaluation dimension (every step found from proposal distribution is multiplied by this quantity)
// This is also used to define the covariance matrix for the Gaussian proposal
double SPEED[] = {1e-1,5e-16,4e-16};
// Mean value for Gaussian proposal (PROPOSAL=1)
double MEAN_P1[] = {0.,0.,0.};
// Number of points in every chain
int N = 1e6;


// PHYSICAL VARIABLES
// Choose to use the constraint on Ct2 in the prior
int CT2_CONSTR = 1;
// Choose the proposal distribution
int PROPOSAL = 1;

// Note that the constraints are defined in rho(pos)



// PROGRAM

typedef struct chain {
  gsl_vector * mean;
  gsl_matrix * covariance;
  double acceptance_ratio;
} chain_res;


inline double cT2 (gsl_vector * pos) {

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  return (2.0-2.0*a)/(2.0-2.0*a+b);

}

inline double cS2 (gsl_vector * pos) {

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  return (b-c)*(2.0*a-2.0)/(2.0*a-b-2.0)/(4.0-4.0*a-b+3.0*c);

}

inline double MPl2 (gsl_vector * pos) {

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  return 4.0-4.0*a-b+3.0*c;

}


inline double rho(gsl_vector * pos){

  double a=gsl_vector_get(pos,0), b=gsl_vector_get(pos,1), c=gsl_vector_get(pos,2);

  // Value of the sound speed squared
  double cs2=cS2(pos);
  // Value of the tensor speed squared
  double ct2=cT2(pos);
  // Value of the Planck mass rescaling factor
  double Mpl2=MPl2(pos);

  // Value of the argument of the Gaussian in the Likelihood
  double delta=fabs(sqrt(ct2)-1.0);

  // This is the experimental sigma on cT2
  double sigma = 4.5e-16;

  double result = 0.;

  // With these ifs I'm introducing the constraints on cT2, cS2 and on the parameters a and c
  // To eliminate these constraints, put CT2_CONSTR=0 at the beginning of the file
  //
  // OLD constraint: CT2_CONSTR==1 && a>=-1. && a<=1. && b>=0. && b<=1. && ct2>=0.0 && ct2<= 1.0 && cs2>=0.0 && cs2<=1.0
  //
  // NEW (correct) constraint: CT2_CONSTR==1 && a>=-1. && a<=1. && Mpl2>=0. && c<=1. && c>=0 && ct2>=0.0 && ct2<= 1.0 && cs2<=0.0
  //
  if (CT2_CONSTR==1 && a>=-1. && a<=1. && c>=0. && c<=1. && ct2>=0. && ct2<=1. && cs2>=0. && cs2<=1.){
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

void C_Delta(double (* chain)[3], chain_res results, double f_stop, int chain_number){

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

  sprintf(filename,"Ligo_abc_DeltaC_t%d.csv",chain_number);
  fp = fopen (filename, "w+");

  for(i=0;i<stop;i++){
    fprintf(fp, "%d,%.8e,%.8e,%.8e\n", i, C_Delta_X[i], C_Delta_Y[i], C_Delta_Z[i]);
  }

  fclose(fp);

  free(C_Delta_X);
  free(C_Delta_Y);
  free(C_Delta_Z);

}

chain_res LHSampling(int proposal, int chain_number){

  int k;

  int id_thread=omp_get_thread_num();

  const gsl_rng_type * T;
  gsl_rng * r;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  unsigned long seed = time(NULL) + chain_number;

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
    sprintf(filename,"Ligo_abc_%d.txt",chain_number);
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

  result.acceptance_ratio = (double) j/i;

  if(DEBUG==1 && result.acceptance_ratio<0.05) printf("\n WARNING (chain %d): Acceptance ratio at %e. The chain seems stuck at initial point. \n", chain_number, result.acceptance_ratio);
  if(DEBUG==1 && result.acceptance_ratio>0.8) printf("\n WARNING (chain %d): Acceptance ratio at %e. The chain seems evolving too slowly. \n", chain_number, result.acceptance_ratio);

  // Check of correlation (with lag)
  if(LAG_TEST==1){

    printf(" Computing Lag function..\n");
    if(DEBUG==1) printf(" Note: This computation can take a lot of computational time.\n");
    if(DEBUG==1) printf("       Lower LAG_TEST_STOP_RATIO for faster results.\n");
    C_Delta(chain,result,LAG_TEST_STOP_RATIO, chain_number);

  }

  #if CSV==1
    if(CSV_COV==1){
      FILE *fp2;
      sprintf(filename,"Ligo_abc_%d.covmat",chain_number);
      fp2 = fopen (filename, "w+");
      fprintf(fp2, "%.8e %.8e %.8e\n", gsl_matrix_get(result.covariance,0,0),gsl_matrix_get(result.covariance,0,1),gsl_matrix_get(result.covariance,0,2));
      fprintf(fp2, "%.8e %.8e %.8e\n", gsl_matrix_get(result.covariance,1,0),gsl_matrix_get(result.covariance,1,1),gsl_matrix_get(result.covariance,1,2));
      fprintf(fp2, "%.8e %.8e %.8e\n", gsl_matrix_get(result.covariance,2,0),gsl_matrix_get(result.covariance,2,1),gsl_matrix_get(result.covariance,2,2));
      fclose(fp2);
    }
  #endif

  #if CSV==1
    if(CSV_CORR==1){
      FILE *fp3;
      sprintf(filename,"Ligo_abc_%d.corr",chain_number);
      fp3 = fopen (filename, "w+");
      fprintf(fp3, "%.8e %.8e %.8e\n", 1.,gsl_matrix_get(result.covariance,0,1)/sqrt(fabs(gsl_matrix_get(result.covariance,0,0)))/sqrt(fabs(gsl_matrix_get(result.covariance,1,1))),gsl_matrix_get(result.covariance,0,2)/sqrt(fabs(gsl_matrix_get(result.covariance,0,0)))/sqrt(fabs(gsl_matrix_get(result.covariance,2,2))));
      fprintf(fp3, "%.8e %.8e %.8e\n", gsl_matrix_get(result.covariance,1,0)/sqrt(fabs(gsl_matrix_get(result.covariance,0,0)))/sqrt(fabs(gsl_matrix_get(result.covariance,1,1))),1.,gsl_matrix_get(result.covariance,1,2)/sqrt(fabs(gsl_matrix_get(result.covariance,1,1)))/sqrt(fabs(gsl_matrix_get(result.covariance,2,2))));
      fprintf(fp3, "%.8e %.8e %.8e\n", gsl_matrix_get(result.covariance,2,0)/sqrt(fabs(gsl_matrix_get(result.covariance,0,0)))/sqrt(fabs(gsl_matrix_get(result.covariance,2,2))),gsl_matrix_get(result.covariance,2,1)/sqrt(fabs(gsl_matrix_get(result.covariance,1,1)))/sqrt(fabs(gsl_matrix_get(result.covariance,2,2))),1.);
      fclose(fp3);
    }
  #endif


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

  int noc = N_CHAINS;

  return sqrt(((N-1.)/N*sigma2_chain + (noc+1.)/N/noc*sigma2_mean)/sigma2_chain);

}

gsl_vector * GR_Test(chain_res * chain_results) {

  int i, k;
  int noc = N_CHAINS;

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

  double R_a = R(gsl_vector_get(sigma2_chain,0), gsl_vector_get(sigma2_mean,0));
  double R_b = R(gsl_vector_get(sigma2_chain,1), gsl_vector_get(sigma2_mean,1));
  double R_c = R(gsl_vector_get(sigma2_chain,2), gsl_vector_get(sigma2_mean,2));

  printf("\n Gelman-Rubin test.\n");
  printf(" \t- R_a : %f\n", R_a);
  printf(" \t- R_b : %f\n", R_b);
  printf(" \t- R_c : %f\n", R_c);

  if(DEBUG==1) printf(" Converge if (approximately) R < 1.2. For better results R < 1.01.\n");

  printf("\n ------------------------------------------------------------------------ \n");

  gsl_vector_free(mean);
  gsl_vector_free(sigma2_chain);
  gsl_vector_free(sigma2_mean);

  gsl_vector * result = gsl_vector_alloc(3);
  gsl_vector_set(result,0,R_a);gsl_vector_set(result,1,R_b);gsl_vector_set(result,2,R_c);

  return result;

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
    printf(" \t- Each of the %d chains will be %d points long.\n", N_CHAINS, N);
  }
  printf(" ------------------------------------------------------------------------ \n");

}

int main(){

  int proposal=PROPOSAL;

  printf(" ------------------------------------------------------------------------ \n");
  printf("\t       mimetic_bayes program ");
  if(DEBUG==1) printf("(with DEBUG mode on)");
  printf("\n");
  printf(" ------------------------------------------------------------------------ \n");

  if(OPENMP==1){

    int i;

    chain_res * chain_results;
    chain_results = (chain_res *) malloc(sizeof(chain_res) * N_CHAINS);

    if(OMP_THREAD_NUM==0) OMP_THREAD_NUM = omp_get_max_threads();
    if(DEBUG==1) {
      printf(" Info on OpenMP on this computer. \n");
      printf(" \t- Number of processors available = %d\n", omp_get_num_procs ( ) );
      printf(" \t- Number of threads              = %d\n", omp_get_max_threads ( ) );
      printf(" \t- Number of working threads      = %d\n", OMP_THREAD_NUM );
      printf(" Specify the number of working threads in .c file.\n");
      printf(" ------------------------------------------------------------------------ \n");
    }

    printf(" Number of chains requested = %d. \n", N_CHAINS);

    printf(" ------------------------------------------------------------------------ \n");

    if(OMP_THREAD_NUM>omp_get_max_threads()){
      printf(" Fatal Error. Can't set number of threads (OMP_THREAD_NUM = %d) bigger than the available number of threads (%d). \n",OMP_THREAD_NUM, omp_get_max_threads());
      printf(" \t Change OMP_THREAD_NUM and try again. \n");
      printf(" The program will be terminated. \n");
      exit(0);
    }

    omp_set_num_threads(OMP_THREAD_NUM);


    init_msg(proposal);

    printf("\n Doing parallelized workflow. Please wait. \n\n");
    if(DEBUG==1) printf(" NOTE: Some WARNING might be shown. \n");
    if(DEBUG==1) printf("       Consider to stop the program and check the parameters if WARNINGs appear. \n\n");

    #pragma omp parallel for
    for(i=1;i<=N_CHAINS;i++)
    {
      int current_thread = omp_get_thread_num();
      if(DEBUG==1) printf(" Thread %d working on chain %d .\n", current_thread, i);
      chain_results[i-1]=LHSampling(proposal, i);
      if(DEBUG==1) printf(" Thread %d finished working on chain %d. \n", current_thread, i);
    }

    printf(" ------------------------------------------------------------------------ \n");

    for(i=0;i<N_CHAINS;i++){

      printf(" CHAIN %d RESULTS. \n", i+1);

      printf("\n Acceptance ratio: %f \n\n", chain_results[i].acceptance_ratio);
      printf(" Results: Mean value pm sqrt(covariant_ii).\n");
      printf(" \ta: %e pm %e\n", gsl_vector_get(chain_results[i].mean,0), sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,0,0))));
      printf(" \tb: %e pm %e\n", gsl_vector_get(chain_results[i].mean,1), sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,1,1))));
      printf(" \tc: %e pm %e\n", gsl_vector_get(chain_results[i].mean,2), sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,2,2))));

      printf("\n Results: Correlation coefficients.\n");
      printf(" \tab: %e \n", gsl_matrix_get(chain_results[i].covariance,0,1)/sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,0,0)))/sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,1,1))));
      printf(" \tac: %e \n", gsl_matrix_get(chain_results[i].covariance,0,2)/sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,0,0)))/sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,2,2))));
      printf(" \tbc: %e \n", gsl_matrix_get(chain_results[i].covariance,1,2)/sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,1,1)))/sqrt(fabs(gsl_matrix_get(chain_results[i].covariance,2,2))));
      printf(" These are the Pearson coefficients : covariant_ij / sigma_i sigma_j\n\n");

      printf(" ------------------------------------------------------------------------ \n");

    }

    gsl_vector * GR_test_result = gsl_vector_alloc(3);
    GR_test_result = GR_Test(chain_results);
    double R_a = gsl_vector_get(GR_test_result,0);
    double R_b = gsl_vector_get(GR_test_result,1);
    double R_c = gsl_vector_get(GR_test_result,2);

    if(DEBUG==1){

      int k = 0;

      double C00 = sqrt(gsl_matrix_get(chain_results[0].covariance,0,0));
      double C11 = sqrt(gsl_matrix_get(chain_results[0].covariance,1,1));
      double C22 = sqrt(gsl_matrix_get(chain_results[0].covariance,2,2));

      printf("\n INFORMATIONS for better convergence. \n\n");
      printf(" WARNING: These are hints based on the results and thumb rules. \n");
      printf("          Follow these hints at your own risk. \n\n");

      if(R_a>1.2||R_b>1.2||R_c>1.2) {
        printf(" - An R_i>1.2 : Chains are sampling different parameter spaces.\n");
        printf("                There might be some troubles in your SPEED vector definition.\n");
        k++;
      }
      if((R_a<1.2&&R_a>1.01)||(R_b<1.2&&R_b>1.01)||(R_c<1.2&&R_c>1.01)) {
        printf(" - An R_i>1.01 : The results might be good, but the results can be improved.\n");
        printf("                 Try different values for the SPEED vector to achieve better results.\n");
        k++;
      }

      double p_value = 0.5;
      if(fabs(SPEED[0]-C00/4.)/fabs(SPEED[0]+C00/4.)>p_value){
        printf(" - From covariance matrix: Try with a SPEED[0] = %e .\n",C00/4.);
        k++;
      }
      if(fabs(SPEED[1]-C11/4.)/fabs(SPEED[1]+C11/4.)>p_value){
        printf(" - From covariance matrix: Try with a SPEED[1] = %e .\n",C11/4.);
        k++;
      }
      if(fabs(SPEED[2]-C22/4.)/fabs(SPEED[2]+C22/4.)>p_value){
        printf(" - From covariance matrix: Try with a SPEED[2] = %e .\n",C22/4.);
        k++;
      }

      int j=0, r=0;
      for(i=0;i<N_CHAINS;i++){
        if(chain_results[i].acceptance_ratio>0.5) j++;
        if(chain_results[i].acceptance_ratio<0.1) r++;
      }
      if(j>0){
        printf(" - From acceptance rations: Maybe some acceptance ratios are too big.\n");
        printf("                            The configuration space might be explored slowly.\n");
        printf("                            This is not necessarily a problem,\n");
        printf("                            but if the results are not good:\n");
        printf("                            try with different SPEED vector values.\n");
        k++;
      }
      if(r>0){
        printf(" - From acceptance rations: Maybe some acceptance ratios are too small.\n");
        printf("                            The configuration space might be explored very inefficiently.\n");
        printf("                            This is not necessarily a problem,\n");
        printf("                            but if the results are not good:\n");
        printf("                            try with different SPEED vector values.\n");
        k++;
      }

      double mean_a = gsl_vector_get(chain_results[0].mean,0);
      double sigma_a = sqrt(fabs(gsl_matrix_get(chain_results[0].covariance,0,0)));
      if(PROPOSAL==1 && (fabs(mean_a-MEAN_P1[0])/fabs(mean_a+MEAN_P1[0])>3.*sigma_a)) {
        printf(" - A burn-in removal might be needed: Try with MEAN_P1[0] = %e.\n",mean_a);
        printf("   Note that this can improve but also worsen the results.\n");
        printf("   Revert the result and make a burn-in removal instead if the results get worse.\n");
      }
      double mean_b = gsl_vector_get(chain_results[0].mean,1);
      double sigma_b = sqrt(fabs(gsl_matrix_get(chain_results[0].covariance,1,1)));
      if(PROPOSAL==1 && (fabs(mean_b-MEAN_P1[1])/fabs(mean_b+MEAN_P1[1])>3.*sigma_b)) {
        printf(" - A burn-in removal might be needed: Try with MEAN_P1[1] = %e.\n",mean_b);
        printf("   Note that this can improve but also worsen the results.\n");
        printf("   Revert the result and make a burn-in removal instead if the results get worse.\n");
      }
      double mean_c = gsl_vector_get(chain_results[0].mean,2);
      double sigma_c = sqrt(fabs(gsl_matrix_get(chain_results[0].covariance,2,2)));
      if(PROPOSAL==1 && (fabs(mean_a-MEAN_P1[2])/fabs(mean_a+MEAN_P1[2])>3.*sigma_c)) {
        printf(" - A burn-in removal might be needed: Try with MEAN_P1[2] = %e .\n",mean_c);
        printf("   Note that this can improve but also worsen the results.\n");
        printf("   Revert the result and make a burn-in removal instead if the results get worse.\n");
      }

      if(k==0){
        printf(" Congratulations! The convercence seems good. \n");
      }

      printf("\n ------------------------------------------------------------------------ \n");

    }

    free(chain_results);

  }
  else{

    init_msg(proposal);

    LHSampling(proposal, 0);
    printf(" ------------------------------------------------------------------------ \n");

  }

  return 0;

}
