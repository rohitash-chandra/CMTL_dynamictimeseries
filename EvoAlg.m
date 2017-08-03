
classdef EvoAlg
    properties
      Individual  % vector of variables
      Population %vector of Individual
      Fitness % Fitness of Individual  
      PopSize  % Number of Individual 
      Problem % Problem Number (Rosenbrock, Sphere or others)
    
      FinalSolution;
      FinalFitness;
      FinalFitIndex;
      FE; % total number of Function Evaluations

      PopMat; 
      
      N;               % number of  variables/problem dimension
       
        
      xmean;    % objective variables initial point
      sigma;          % coordinate wise standard deviation (step size)
  
      mu;               % number of parents/points for recombination
      weights; % muXone array for weighted recombination       
      mueff; % variance-effectiveness of sum w_i x_i

      cc; % time constant for cumulation for C
      cs;  % t-const for cumulation for sigma control
      c1;    % learning rate for rank-one update of C
      cmu;  % and for rank-mu update
      damps; % damping for sigma 
                                                      % usually close to 1
  % Initialize dynamic (internal) strategy parameters and constants
      pc;   % evolution paths for C and sigma
      ps;
      B;                       % B defines the coordinate system
      D;                      % diagonal D defines the scaling
      C;            % covariance matrix C
      invsqrtC;    % C^-1/2 
      eigeneval;                      % track update of B and D
      chiN;  % expectation of 
      
    end
   
   methods (Static)
       
        function SolVec = GetSolution(obj)
         SolVec = obj.FinalSolution;
         end
      
        function BestFit = GetFitness(obj)
         BestFit = obj.FinalFitness; 
     
        end
        
        function TotalFE = GetFE(obj)
         TotalFE = obj.FE;
        end
       
          function obj = EvoAlg( PopSize,Dimen, Prob, ProbMax, ProbMin)
           obj.FE = 0;
           obj.PopSize = PopSize; 
           obj.Problem = Prob;
           obj.FinalFitness = 20004; 
          obj.FinalFitIndex = 1;
          
          max_ =  ProbMax(1);
          min_ =  ProbMin(1);
          
          
            obj.N = Dimen;               % number of objective variables/problem dimension
            
          %----------------------------------------------------------------   
          for i = 1:obj.PopSize 
              obj.Population(i).Individual = ((max_-min_).*rand(obj.N,1) + min_)' ;
         end
          
          obj.PopMat = ((max_-min_).*rand( obj.N,obj.PopSize)+ min_);
        
           
            obj.xmean =((max_-min_).* rand(obj.N,1)+ min_);    % objective variables initial point
            obj.sigma = 0.5;          % coordinate wise standard deviation (step size)
  
            obj.mu = obj.PopSize/2;               % number of parents/points for recombination
            obj.weights = log(obj.mu+1/2)-log(1:obj.mu)'; % muXone array for weighted recombination
            obj.mu = floor(obj.mu);        
            obj.weights = obj.weights/sum(obj.weights);     % normalize recombination weights array
            obj.mueff=sum(obj.weights)^2/sum(obj.weights.^2); % variance-effectiveness of sum w_i x_i

 	     % Strategy parameter setting: Adaptation
 	    obj.cc = (4 + obj.mueff/obj.N) / (obj.N+4 + 2*obj.mueff/obj.N); % time constant for cumulation for C
 	    obj.cs = (obj.mueff+2) / (obj.N+obj.mueff+5);  % t-const for cumulation for sigma control
  	    obj.c1 = 2 / ((obj.N+1.3)^2+obj.mueff);    % learning rate for rank-one update of C
 	    obj.cmu = min(1-obj.c1, 2 * (obj.mueff-2+1/obj.mueff) / ((obj.N+2)^2+obj.mueff));  % and for rank-mu update
  	    obj.damps = 1 + 2*max(0, sqrt((obj.mueff-1)/(obj.N+1))-1) + obj.cs; % damping for sigma 
                                                      % usually close to 1
  	    % Initialize dynamic (internal) strategy parameters and constants
  	    obj.pc = zeros(obj.N,1);
  	    obj.ps = zeros(obj.N,1);   % evolution paths for C and sigma
  	    obj.B = eye(obj.N,obj.N);                       % B defines the coordinate system
  	    obj.D = ones(obj.N,1);                      % diagonal D defines the scaling
  	    obj.C = obj.B * diag(obj.D.^2) * obj.B';            % covariance matrix C
  	    obj.invsqrtC = obj.B * diag(obj.D.^-1) * obj.B';    % C^-1/2 
 	    obj.eigeneval = 0;                      % track update of B and D
  	    obj.chiN=obj.N^0.5*(1-1/(4*obj.N)+1/(21*obj.N^2));  % expectation of 
          
          
      end 
       
      function PrintGA(obj)
   
          obj.PopSize
          obj.NumDimen
         obj.Problem 
      end
       

    % --------------------------------------------------------------------------------------- 
   function  obj =PureCMAES(obj, stopeval,sp, CCIndividual,CCDimen, CCDimenIndex,  TaskTopo,  net) 
            % sp
%             
%                net{1}
%                 net{2}
%                  net{3}
%                   net{4}
           
      counteval = 0;  % the next 40 lines contain the 20 lines of interesting code 
       while counteval < stopeval
      
            for k=1:obj.PopSize 
               obj.PopMat(:,k) =   obj.xmean + obj.sigma  * obj.B'  * (obj.D .* randn(obj.N,1)); % m + sig * Normal(0,C) 
               counteval = counteval+1;
            end
        
              for i = 1:obj.PopSize  
              Sol =  obj.PopMat(:,i)';
               obj.Population(i).Individual = Sol; 
              end 
                
              for k=1:obj.PopSize
          
                 Solution =  obj.PopMat(:,k);
                  SolSize =   size(Solution);
                 Ind = CCIndividual;
               
                 for i=1:SolSize 
                      Ind( CCDimenIndex(sp)+i) = Solution(i);
                 end
                % Ind 
                 
               obj.Fitness(k)  = FNNetwork.EvaluateNNSol(net{sp}, Ind,  TaskTopo,   sp);  
            %   obj.Fitness(k) = rand;
          end
              %obj.Fitness    = fcnsuite_func( obj.PopMat, obj.Problem);
   
        % Sort by fitness and compute weighted mean into xmean
            [obj.Fitness, arindex] = sort(obj.Fitness);  % minimizationc
             xold = obj.xmean;
             obj.xmean = obj.PopMat(:,arindex(1:obj.mu)) * obj.weights;  % recombination, new mean value
    
    % Cumulation: Update evolution paths
             obj.ps = (1-obj.cs) * obj.ps  + sqrt(obj.cs*(2-obj.cs)*obj.mueff) * obj.invsqrtC * (obj.xmean-xold) / obj.sigma; 
             hsig = sum(obj.ps.^2)/(1-(1-obj.cs)^(2*counteval/obj.PopSize))/obj.N < 2 + 4/(obj.N+1);
              obj.pc = (1-obj.cc) * obj.pc + hsig * sqrt(obj.cc*(2-obj.cc)*obj.mueff) * (obj.xmean-xold) / obj.sigma; 

    % Adapt covariance matrix C
             artmp = (1/obj.sigma) * (obj.PopMat(:,arindex(1:obj.mu)) - repmat(xold,1,obj.mu));  % mu difference vectors
             obj.C = (1-obj.c1-obj.cmu) * obj.C   + obj.c1 * (obj.pc * obj.pc'  + (1-hsig) * obj.cc*(2-obj.cc) * obj.C) ... % minor correction if hsig==0
                + obj.cmu * artmp * diag(obj.weights) * artmp'; % plus rank mu update 

    % Adapt step size sigma
            obj.sigma = obj.sigma * exp((obj.cs/obj.damps)*(norm(obj.ps)/obj.chiN - 1)); 
    
    % Update B and D from C
    if counteval - obj.eigeneval > obj.PopSize/(obj.c1+obj.cmu)/obj.N/10  % to achieve O(N^2)
      obj.eigeneval = counteval;
      obj.C = triu(obj.C) + triu(obj.C,1)'; % enforce symmetry
      [obj.B,obj.D] = eig(obj.C);           % eigen decomposition, B==normalized eigenvectors
      obj.D = sqrt(diag(obj.D));        % D contains standard deviations now
      obj.invsqrtC = obj.B * diag(obj.D.^-1) * obj.B';
    end
    
    % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable 
    %if arfitness(1) <= stopfitness || max(D) > 1e7 * min(D)
     % break;
   % end
   
   
  
   end % while, end generation loop

   obj.FinalSolution= obj.PopMat(:, arindex(1)); % Return best point of last iteration. 
    obj.FinalFitIndex = arindex(1) ;
   obj.FinalFitness = obj.Fitness(arindex(1)); % Return best point of last iteration.
 
  
end
      
      % main evolution
      % -------------------------------------------------------------------------
%       function obj = Evolution(obj, MaxFE,   MinError  )
                %  Gen = 1;
%           while obj.FE < 1
                 
%               obj = EvoAlg.PureCMAES(obj, MaxFE, 1)  ;
                
%               obj.FE = obj.FE + obj.PopSize ;
 
%                  Gen= Gen +1;
%               end 

%                obj.FinalFitness =obj.Fitness(1);
% 		            obj.FinalFitIndex =   1;
%                    %obj.FinalSolution = obj.Population(obj.FinalFitIndex).Individual;
%               
%               obj.FinalSolution 
%               obj.FinalFitness
%               
%               
%       end
            
          
      end
   end
    
%end
