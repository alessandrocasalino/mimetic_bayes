// Compile with: gcc-7 -O2 Ligo_abc.c -o Ligo_abc.exe -lgsl -lgslcblas -fopenmp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <sys/time.h>
#include <omp.h>


// Variables for parallelization with OpenMP
// Info on OpenMP: https://bisqwit.iki.fi/story/howto/openmp/#Syntax
int OPENMP = 1;
int OMP_THREAD_NUM = 8;

// Variables for the output of the program
int LAG_TEST = 0;
int HPD_COMP = 0;
int CT2_CONSTR = 1;
int CSV = 0;



// Variables for the MCMC evaluation
double X_START = 0.;
double Y_START = 0.;
double Z_START = 0.;
double X_SPEED = 1.;
double Y_SPEED = 1e-15;
double Z_SPEED = 1.;
int N = 1e6;






typedef struct position {
  double x;
  double y;
  double z;
} t_pos;

typedef struct covariance {
  double XX;
  double YY;
  double ZZ;
  double XY;
  double XZ;
  double YZ;
} cov_mat;

typedef struct chain {
  t_pos mean;
  cov_mat covariance;
} chain_res;

inline double cT2 (t_pos pos) {
  double a=pos.x, b=pos.y;
  return (2.0-2.0*a)/(2.0-2.0*a+b);
}

inline double rho(t_pos pos){

  double a=pos.x, b=pos.y, c=pos.z;

  // Value of the sound speed squared
  double cS2=(b-c)*(2.0*a-2.0)/(2.0*a-b-2.0)/(4.0-4.0*a-b-3.0*c);
  // Value of the tensor speed squared
  double cT2=(2.0-2.0*a)/(2.0-2.0*a+b);
  // Value of the argument of the Gaussian in the Likelihood
  double delta=fabs(sqrt(cT2)-1.0);

  // This is the experimental sigma on cT2
  double sigma = 4.5e-16;
  double result = 0.;

  // With these ifs I'm introducing the constraints on cT2, cS2 and on the parameters a and c
  // To eliminate these constraints, put CT2_CONSTR=0 at the beginning of the file
  if (CT2_CONSTR==1 && pos.x>=-1. && pos.x<=1. && pos.z>=-4.0/3.0 && pos.z<=0. && cT2>=0.0 && cT2<= 1.0 && cS2>=0.0 && cS2<=1.0){
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

double mean(double * vec){

  double mean=0.;

  int i;
  for(i=0;i<N;i++){
    mean=mean+vec[i];
  }
  return mean/N;
}

cov_mat covariance_matrix(double * vec_X, double * vec_Y, double * vec_Z){

  cov_mat cm = {0.,0.,0.,0.,0.,0.};

  double mean_X = mean(vec_X);
  double mean_Y = mean(vec_Y);
  double mean_Z = mean(vec_Z);

  int i;
  for(i=0;i<N;i++){
    cm.XX=cm.XX+(vec_X[i]-mean_X)*(vec_X[i]-mean_X);
    cm.YY=cm.YY+(vec_Y[i]-mean_Y)*(vec_Y[i]-mean_Y);
    cm.ZZ=cm.ZZ+(vec_Z[i]-mean_Z)*(vec_Z[i]-mean_Z);
    cm.XY=cm.XY+(vec_X[i]-mean_X)*(vec_Y[i]-mean_Y);
    cm.XZ=cm.XZ+(vec_X[i]-mean_X)*(vec_Z[i]-mean_Z);
    cm.YZ=cm.YZ+(vec_Y[i]-mean_Y)*(vec_Z[i]-mean_Z);
  }

  cm.XX=cm.XX/(N-1.);
  cm.YY=cm.YY/(N-1.);
  cm.ZZ=cm.ZZ/(N-1.);
  cm.XY=cm.XY/(N-1.);
  cm.XZ=cm.XZ/(N-1.);
  cm.YZ=cm.YZ/(N-1.);

  return cm;

}

double Gaussian_proposal(double mean, double sigma, double u1, double u2) {

  // Box Muller transform from flat distribution to Guassian distribution
  return mean + sigma * sqrt(-2.*log(u1)) * cos(2.*M_PI*u2);

}

void C_Delta(double * vec_X, double * vec_Y, double * vec_Z, cov_mat cm, double f_stop){

  int id_thread=omp_get_thread_num();

  double mean_X = mean(vec_X);
  double mean_Y = mean(vec_Y);
  double mean_Z = mean(vec_Z);

  double * C_Delta_X; double * C_Delta_Y; double * C_Delta_Z;
  C_Delta_X = (double *) malloc(sizeof(double) * N);
  C_Delta_Y = (double *) malloc(sizeof(double) * N);
  C_Delta_Z = (double *) malloc(sizeof(double) * N);

  int stop = N*f_stop;

  int i,j;
  for(i=0;i<stop;i++){
    for(j=0;j<stop-i;j++){
      C_Delta_X[i]=C_Delta_X[i]+(vec_X[j]-mean_X)*(vec_X[j+i]-mean_X);
      C_Delta_Y[i]=C_Delta_Y[i]+(vec_Y[j]-mean_Y)*(vec_Y[j+i]-mean_Y);
      C_Delta_Z[i]=C_Delta_Z[i]+(vec_Z[j]-mean_Z)*(vec_Z[j+i]-mean_Z);
    }
    C_Delta_X[i]=C_Delta_X[i]/(N-i)/cm.XX;
    C_Delta_Y[i]=C_Delta_Y[i]/(N-i)/cm.YY;
    C_Delta_Z[i]=C_Delta_Z[i]/(N-i)/cm.ZZ;
  }

  FILE *fp;
  char filename[25];

  sprintf(filename,"Ligo_abc_Chain_t%d.csv",id_thread);
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

  int id_thread=omp_get_thread_num();

  const gsl_rng_type * T;
  gsl_rng * r;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  unsigned long seed = time(NULL) + id_thread;

  gsl_rng_set(r, seed);

  t_pos pos = {X_START, Y_START, Z_START};

  FILE *fp;
  FILE *fp_1s;
  FILE *fp_2s;
  char filename[25];

  if(CSV==1){
    sprintf(filename,"Ligo_abc_Chain_t%d.csv",id_thread);
    fp = fopen (filename, "w+");
    if(HPD_COMP==1){
      sprintf(filename,"Ligo_abc_Chain_1s_t%d.csv",id_thread);
      fp_1s = fopen (filename, "w+");
      sprintf(filename,"Ligo_abc_Chain_2s_t%d.csv",id_thread);
      fp_2s = fopen (filename, "w+");
    }
  }

  double * vec_a; double * vec_b; double * vec_c;
  vec_a = (double *) malloc(sizeof(double) * N);
  vec_b = (double *) malloc(sizeof(double) * N);
  vec_c = (double *) malloc(sizeof(double) * N);

  int i = 0;
  int j = 0;

  while (i<N) {

    double f0 = rho(pos);

    t_pos pos_disp;
    t_pos pos_temp;

    // NOTE: Introducing the flat proposal distribution here
    // Modify here to introduce another proposal distribution

    if(proposal==0){

      double random_number = 2.0 * gsl_rng_uniform (r) - 1.0;
      pos_disp.x = X_SPEED * random_number;

      random_number = 2.0 * gsl_rng_uniform (r) - 1.0;
      pos_disp.y = Y_SPEED * random_number;

      random_number = 2.0 * gsl_rng_uniform (r) - 1.0;
      pos_disp.z = Z_SPEED * random_number;

      pos_temp.x = pos.x + pos_disp.x;
      pos_temp.y = pos.y + pos_disp.y;
      pos_temp.z = pos.z + pos_disp.z;

    }
    else if(proposal=!0){

      cov_mat L = {X_SPEED * 0.4, Y_SPEED, Z_SPEED * 0.3, 0., 0., 0.};
      t_pos gauss_mean = {X_START, Y_START, Z_START};

      t_pos pos_pp;

      pos_pp.x = Gaussian_proposal(0., 1., gsl_rng_uniform (r), gsl_rng_uniform (r));
      pos_pp.y = Gaussian_proposal(0., 1., gsl_rng_uniform (r), gsl_rng_uniform (r));
      pos_pp.z = Gaussian_proposal(0., 1., gsl_rng_uniform (r), gsl_rng_uniform (r));

      pos_disp.x = gauss_mean.x + sqrt(fabs(L.XX)) * pos_pp.x + sqrt(fabs(L.XY)) * pos_pp.y + sqrt(fabs(L.XZ)) * pos_pp.z;
      pos_disp.y = gauss_mean.y + sqrt(fabs(L.XY)) * pos_pp.x + sqrt(fabs(L.YY)) * pos_pp.y + sqrt(fabs(L.YZ)) * pos_pp.z;
      pos_disp.z = gauss_mean.z + sqrt(fabs(L.XZ)) * pos_pp.x + sqrt(fabs(L.YZ)) * pos_pp.y + sqrt(fabs(L.ZZ)) * pos_pp.z;

      pos_disp.x = gauss_mean.x + L.XX * pos_pp.x + L.XY * pos_pp.y + L.XZ * pos_pp.z;
      pos_disp.y = gauss_mean.y + L.XY * pos_pp.x + L.YY * pos_pp.y + L.YZ * pos_pp.z;
      pos_disp.z = gauss_mean.z + L.XZ * pos_pp.x + L.YZ * pos_pp.y + L.ZZ * pos_pp.z;

      pos_temp.x = pos.x + pos_disp.x;
      pos_temp.y = pos.y + pos_disp.y;
      pos_temp.z = pos.z + pos_disp.z;

    }

    double f1 = rho(pos_temp);

    if (f1>f0) {
      pos = pos_temp;
      j++;
    }
    else {

      double ra = gsl_rng_uniform (r);
      if (ra<f1/f0) {
        pos = pos_temp;
        j++;
      }
    }

    vec_a[i]=pos.x;
    vec_b[i]=pos.y;
    vec_c[i]=pos.z;

    if(CSV==1){

      double rho_val=rho(pos);
      double cT2_val=cT2(pos);

      fprintf(fp, "%d,%.24e,%.24e,%.24e,%.24e,%.24e\n", i, pos.x, pos.y, pos.z, rho_val, cT2_val);
      if(HPD_COMP==1){

        t_pos sigma_value = {0.6827,0.9545,0.99994};

        int i,k; int j=0;
        if(rho_val>=1.0-sigma_value.x){
          fprintf(fp_1s, "%.24e,%.24e,%.24e,%.24e,%.24e\n", pos.x, pos.y, pos.z, rho_val, cT2_val);
        }
        else if(rho_val>=1.0-sigma_value.y && rho_val<1.0-sigma_value.x){
          fprintf(fp_2s, "%.24e,%.24e,%.24e,%.24e,%.24e\n", pos.x, pos.y, pos.z, rho_val, cT2_val);
        }
      }
    }

    i++;

  }

  if(CSV==1){
    fclose(fp);
    if(HPD_COMP==1){
      fclose(fp_1s);
      fclose(fp_2s);
    }
  }

  chain_res result;

  result.mean.x = mean(vec_a);
  result.mean.y = mean(vec_b);
  result.mean.z = mean(vec_c);
  result.covariance = covariance_matrix(vec_a,vec_b,vec_c);

  printf("\n Acceptance ratio: %f \n\n", (double) j/i);
  printf(" Results: mean value pm sqrt(covariant_ii)\n");
  printf(" \ta: %e pm %e\n", result.mean.x, sqrt(fabs(result.covariance.XX)));
  printf(" \tb: %e pm %e\n", result.mean.y, sqrt(fabs(result.covariance.YY)));
  printf(" \tc: %e pm %e\n", result.mean.z, sqrt(fabs(result.covariance.ZZ)));

  printf("\n Covariance coefficients (Pearson coefficients : covariant_ij / sigma_i sigma_j): \n");
  printf(" \tAB: %e \n", result.covariance.XY/sqrt(fabs(result.covariance.XX))/sqrt(fabs(result.covariance.YY)));
  printf(" \tAC: %e \n", result.covariance.XZ/sqrt(fabs(result.covariance.XX))/sqrt(fabs(result.covariance.ZZ)));
  printf(" \tBC: %e \n\n", result.covariance.YZ/sqrt(fabs(result.covariance.YY))/sqrt(fabs(result.covariance.ZZ)));

  // Check of correlation (with lag)
  if(LAG_TEST){
    C_Delta(vec_a,vec_b,vec_c,result.covariance,0.1);

  }
  free(vec_a);
  free(vec_b);
  free(vec_c);

  gsl_rng_free (r);

  return result;

}

double R(double sigma2_chain, double sigma2_mean){

  int noc = OMP_THREAD_NUM;

  return sqrt(((N-1.)/N*sigma2_chain + (noc+1.)/N/noc*sigma2_mean)/sigma2_chain);

}

void GR_Test(chain_res * chain_results) {

  int i;
  int noc = OMP_THREAD_NUM;

  t_pos mean = {0., 0., 0.};
  t_pos sigma2_chain = {0., 0., 0.};
  t_pos sigma2_mean = {0., 0., 0.};

  for(i=0;i<noc;i++){

    mean.x = mean.x + chain_results[i].mean.x/noc;
    mean.y = mean.y + chain_results[i].mean.y/noc;
    mean.z = mean.z + chain_results[i].mean.z/noc;

    sigma2_chain.x = sigma2_chain.x + fabs(chain_results[i].covariance.XX)/noc;
    sigma2_chain.y = sigma2_chain.y + fabs(chain_results[i].covariance.YY)/noc;
    sigma2_chain.z = sigma2_chain.z + fabs(chain_results[i].covariance.ZZ)/noc;

  }

  for(i=0;i<noc;i++){

    sigma2_mean.x = sigma2_mean.x + pow(mean.x-chain_results[i].mean.x,2.)*N/(noc-1.);
    sigma2_mean.y = sigma2_mean.y + pow(mean.y-chain_results[i].mean.y,2.)*N/(noc-1.);
    sigma2_mean.z = sigma2_mean.z + pow(mean.z-chain_results[i].mean.z,2.)*N/(noc-1.);

  }

  printf(" Gelman-Rubin test.\n");
  printf(" \t- R_a : %f\n", R(sigma2_chain.x,sigma2_mean.x));
  printf(" \t- R_b : %f\n", R(sigma2_chain.y,sigma2_mean.y));
  printf(" \t- R_c : %f\n", R(sigma2_chain.z,sigma2_mean.z));
  printf(" Converge if (approximately) R < 1.2.\n");
  printf(" -------------------------------------------------- \n");

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

  int proposal=1;

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
