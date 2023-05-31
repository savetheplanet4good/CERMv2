import numpy as np
import copy

from utils import correlation, migmat_to_thresh, thresh_to_migmat, correlation_from_covariance

"""This script defines the class MatrixAndThresholdGenerator, which computes migration matrices and thresholds under a given scenario"""

class MatrixAndThresholdGenerator:
    """
    A class to represent all migration quantities.
    
    Attributes
    ----------
    nb_groups: int
        number of groups in input portfolio
    nb_ratings: int
        number of ratings in input ratings table
    scenarios: ScenarioGenerator
        climate scenario input, of type ScenarioGenerator
    ratings: Ratings
        ratings input, of type Ratings
    loan_profile: list (float)
        loan profile of the bank
    nb_rf: int
        number of risk factors*
    amortization_matrices: array (float)
        amortization matrices (see formula (11))
    amortization: float
        amortization coefficient (see formula (10))
    migration_matrices: array (float)
        migration matrices
    reg_thresholds: array (float)
        regulatory transition thresholds
    thresholds: array (float)
        transition thresholds
    micro_correlation: array (float)
        micro-correlations
    reg_factor_loadings: array (float)
        regulatory factor loadings
    """

    def __init__(self, ratings, groups, scenarios,duration,loan_profile,micro_correlation,transition_target_date, sensitivity_objective="regular"):
        """
        Constructs all the necessary attributes for the MatrixAndThresholdGenerator object.
        
        Parameters
        ----------
            ratings : Ratings
                ratings input, of Ratings type
            groups: list (float)
                groups of portfolio
            scenarios: ScenarioGenerator
                climate scenario input, of ScenarioGenerator type
            loan_profile: list (float)
                loan profile of the bank
        """
        
        import datetime
        today = datetime.date.today()
        year = today.year

        
        self.nb_groups = len(groups)
        self.nb_ratings = len(ratings.list())
        self.scenarios = scenarios
        self.ratings = ratings
        self.sensitivity_objective = sensitivity_objective
        self.loan_profile=loan_profile
        self.duration= duration
        
        self.transition_horizon=transition_target_date-year
        
        self.micro_correlation=micro_correlation
    

        self.init()

    def init(self):
        """
        To initialize migration matrices calculations (structures at time 0).
        """
        nb_groups = self.nb_groups
        nb_ratings = self.nb_ratings
        nb_rf = self.scenarios.nb_rf
        loan_profile=self.loan_profile
        duration=self.duration

        #initialization of initial regulatory migration matrices and thresholds

        tmp_reg_matrix = np.zeros((nb_groups, nb_ratings, nb_ratings))
        thresholds = np.zeros((nb_groups, nb_ratings, nb_ratings))
        for g in range(nb_groups):
            tmp_reg_matrix[g] = self.ratings.reg_mat()
            thresholds[g] = migmat_to_thresh(tmp_reg_matrix[g])

        reg_migration_matrices = copy.deepcopy(tmp_reg_matrix)

        #generation of amortization matrices followingloan profile (see (11) in CERM paper)

        #print(loan_profile)
        
        self.amortization_matrices=np.zeros((self.scenarios.horizon,nb_groups, nb_ratings, nb_ratings))
        self.amortization=1/duration
        ones=np.ones((nb_ratings,1))
        #print(ones)
        ones[7,0]=0
        lp=np.reshape(loan_profile,(3,nb_ratings))
        Transition_rating=np.ones((1,nb_ratings))
        Transition_rating=(lp[0,:]-lp[2,:])/self.transition_horizon
        #print(lp[1].reshape(1,nb_ratings))
        #print(ones)
        
        for t in range(1, self.scenarios.horizon):    

            #transition rating
            if t<=self.transition_horizon:
                lp[2]+=Transition_rating
                
            for g in range(nb_groups):
                self.amortization_matrices[t,g]=ones@lp[g].reshape(1,nb_ratings)
            
        #print(self.amortization_matrices[:])

        #initialization of migration matrices

        self.migration_matrices = np.zeros((self.scenarios.horizon, nb_groups, nb_ratings, nb_ratings))
        self.migration_matrices[0, :, :, :] = copy.deepcopy(tmp_reg_matrix)

        #initialization of thresholds

        self.reg_thresholds = copy.deepcopy(thresholds)
        self.thresholds = np.zeros((self.scenarios.horizon, nb_groups, nb_ratings, nb_ratings))
        self.thresholds[0, :, :, :] = copy.deepcopy(thresholds)

        #generation of micro-correlations, chosen all constant to 1 so far
        ##to be calibrated in the long run

        #self.micro_correlation = np.ones((nb_groups, nb_ratings, nb_rf))
        
        #set micro correlation
        #Brawn by default
        #self.micro_correlation[:,:,1]=1.5
        #self.micro_correlation[:,:,2]=2
        
        #Green
        #self.micro_correlation[0,:,1]=0.5
        #self.micro_correlation[0,:,2]=0
        
        #Adapted
        #self.micro_correlation[1,:,1]=0.5
        
        #Mitigated
        #self.micro_correlation[2,:,2]=0
       
        #Transition from Brown to Green
        #self.transition_horizon=28
    
        self.sensitivity_objective =0.01
        
        
        
        #print("micro correlation")
        #print (self.micro_correlation)

        #accounting for regular transition of micro-correlations towards sensititvity_objective
        
        if self.sensitivity_objective !="regular":
            self.diff=np.zeros((nb_groups, nb_ratings, nb_rf))
            
            print (self.transition_horizon)
            #self.diff = (self.sensitivity_objective-np.ones((nb_groups, nb_ratings, nb_rf)))/self.scenarios.horizon
            #self.diff = (self.sensitivity_objective-np.ones((nb_groups, nb_ratings, nb_rf)))/self.transition_horizon
            #self.diff[1,:,:] = (self.micro_correlation[0,:,:]-self.micro_correlation[1,:,:])/self.transition_horizon
            self.diff[2,:,:] = (self.micro_correlation[0,:,:]-self.micro_correlation[2,:,:])/self.transition_horizon
            #self.diff[:3,:,:]=0
            #self.diff[4,:,0]=0
        else:
            self.diff=0
            
        #print("transition rate")
        #print (self.diff)
        
        #initialization of tilded factor_loadings

        a_tilde = np.zeros((nb_groups, nb_ratings, nb_rf))

        #generation of tilded factor loadings

        for j in range(nb_rf):
            a_tilde[:, :, j] = self.micro_correlation[:, :, j] * self.scenarios.macro_correlation_at(1)[j]

        #initialization of regulatory factor loadings and correlation coefficients

        a_reg = np.zeros((nb_groups, nb_ratings, nb_rf))
        r_reg = np.zeros((nb_groups, nb_ratings))

        #generation of regulatory factor loadings and correlation coefficients
        for g in range(nb_groups):
            for i in range(nb_ratings):
                # correlation model to economic risk
                r_reg[g, i] = correlation(reg_migration_matrices[g][i, nb_ratings - 1])
                a = np.sqrt(r_reg[g, i]) * a_tilde[g, i]
                b = np.sqrt(a_tilde[g, i].T @ (self.scenarios.rf_correlation_at(1) @ a_tilde[g, i]))
                a_reg[g, i] = a / b
        
        #generation of initial regulatory factor loadings and c factor

        self.reg_factor_loadings = copy.deepcopy(a_reg)
        self.init_tilde_factor_loadings = a_tilde
        self.c_factor = copy.deepcopy(a_reg)

        #initialization of factor loadings

        self.factor_loadings = np.zeros((self.scenarios.horizon, nb_groups, nb_ratings, nb_rf))
        self.factor_loadings[0, :, :, :] = copy.deepcopy(a_reg)

        #initialization of ratio2 (calculation intermediary)

        self.ratio2 = np.zeros((self.scenarios.horizon, nb_groups, nb_ratings))

        #generation of time-run migration matrices (product of consecutive migration matrices) 

        self.product = np.zeros((self.nb_groups, self.scenarios.horizon, self.nb_ratings, self.nb_ratings))
        self.product[:,0,:,:] = self.migration_matrices[0,:]

    def compute(self):
        """
        Computes calculations for non-random quantities only
        """
        scenarios = self.scenarios
        horizon = scenarios.horizon 
        nb_groups = self.nb_groups
        nb_ratings = self.nb_ratings
        nb_rf = scenarios.nb_rf

        #initialization of correlation matrix

        corr_at_0 = scenarios.rf_correlation_at(0)

        #execution

        for t in range(1, horizon):

            #update of micro-correlations: step towards sensitivity_objective
            


            if t<=self.transition_horizon:
                self.micro_correlation+=self.diff

            #update of tildes factor loadings

            a_tilde = np.zeros((nb_groups,nb_ratings, nb_rf))
            for j in range(nb_rf):
                a_tilde[:, :, j] = self.micro_correlation[:, :, j] * scenarios.macro_correlation_at(t)[j]

            #update of c factors

            self.c_factor = self.reg_factor_loadings * a_tilde / self.init_tilde_factor_loadings
            self.c_factor[self.init_tilde_factor_loadings == 0] = 0

            #logging of correlation matrix at time t

            corr_at_t = scenarios.rf_correlation_at(t)

            for g in range(self.nb_groups):
                for k in range(self.nb_ratings):

                    #intermediary calculations

                    c_factor = self.c_factor[g, k]
                    reg_factor = self.reg_factor_loadings[g, k]
                    ratio1 = 1 + c_factor @ (corr_at_t @ c_factor.T) - reg_factor @ (corr_at_0 @ reg_factor.T)

                    #update of factor loadings

                    self.factor_loadings[t, g, k] = c_factor / np.sqrt(ratio1)

                    #update of thresholds

                    self.thresholds[t] = self.reg_thresholds / np.sqrt(ratio1)

                    # update of ratios for conditional thresholds

                    self.ratio2[t,g,k] = 1 - self.factor_loadings[t, g, k] @ (corr_at_t @ self.factor_loadings[t, g, k].T)

    def conditional(self):
        """
        Computes calculations for random quantities.
        This allows to optimize computations given that only conditional is executed iteratively with every random climate trajectory.
        """
        scenarios = self.scenarios
        horizon = scenarios.horizon
        nb_groups = self.nb_groups
        nb_ratings = self.nb_ratings

        #logging of systematic risks

        srisks = np.array([np.array(self.factor_loadings[t])@(scenarios.risks_at(t)) for t in range(horizon)])
        
       

        #resizing of systematic risks

        mul = np.array([[1]*nb_ratings])
        srisks_resized = srisks@mul

        #resizing of ratio2

        self.ratio2 = np.resize(self.ratio2, (horizon, nb_groups, nb_ratings, 1))
        self.ratio2_resized = self.ratio2@mul

        #generation of conditional thresholds

        self.conditional_thresholds = (self.thresholds - srisks_resized)/self.ratio2_resized

        #generation of migration matrices conditional to systematic risks

        self.migration_matrices[1:] = thresh_to_migmat(self.conditional_thresholds[1:])

        #adaptation of migration matrices with amortization

        self.migration_matrices = (1-self.amortization)*self.migration_matrices + self.amortization*self.amortization_matrices
        
        self.migration_matrices[:,:,7,7]=1
        
        #print("migration_matrices")
        #print(self.migration_matrices[49,0,7])

        #generation of time-run migration matrices (product of consecutive migration matrices)
        


        for t in range(1, horizon):
          
            for g in range(nb_groups):
                self.product[g,t] = self.product[g,t-1]@self.migration_matrices[t,g]
