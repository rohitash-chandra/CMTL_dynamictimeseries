


% Cooperative Coevolution with CMAES in Sub-Pops by Dr. Rohitash Chandra (2016)
% c.rohitash@gmail.com .  Note that the function call is in the base class
% EvoAlg() which implemented CMAES. 





%http://stackoverflow.com/questions/13426234/array-of-classes-in-matlab



classdef CooperativeCoevolution   < EvoAlg
    properties
      SP;  % vector of subpopulations from GA
      
      Table; % Table of Best Solutions
     
      CCFitness; % Fitness of Individual 
      CCNumDimen;  % Size of Individual
      DimenIndex;
      CCProblem; % Problem Number (Rosenbrock, Sphere or others)
      
      NumSP;
      Success;
      TaskSuccess;

      CCIndividual; % all combined SP best sol
       TaskIndividual; % solution for each task in multitasking environment. this can be combined ind from diff SP
       
      CCFinalFitness;
      CCFE; % total number of Function Evaluations
      CCDimen;
      
      SPFitList; % Vector of Fitness of all SP
      
      Index;
      
    end
   
   methods (Static)
       
        function SolVec = GetSolution(ccobj)
         SolVec = ccobj.CCIndividual;
        end
       function SolVec = SeprateTaskSolution(ccobj, sp)
       SolVec =   ccobj.Index(sp).TaskIndividual;
        end
           

        function S = GetTaskSuccess(ccobj, task)
         S = ccobj.TaskSuccess(task);
        end 

        function S = GetSuccess(ccobj)
         S = ccobj.Success;
        end 
        
        function BestFit = GetFitness(ccobj)
         BestFit = ccobj.CCFinalFitness;
        end
        
        function TotalFE = GetFE(ccobj)
         TotalFE = ccobj.CCFE;
        end
        
        function FitList = GetFitList(ccobj)
         FitList = ccobj.SPFitList ;
        end
        
        
      
        function ccobj = CooperativeCoevolution(PopSize,Dimen, Prob, ProbMax, ProbMin )
            
          ccobj =   ccobj@EvoAlg(PopSize,Dimen(1), Prob, ProbMax, ProbMin); %  this has to be done to inilize the Inheritance - this is not used later as vector(cell) of EA. 
             
           
            ccobj.CCFE = 0;
           ccobj.CCDimen = Dimen;
           ccobj.CCNumDimen = sum(Dimen); 
           ccobj.CCProblem = Prob;
           ccobj.CCFinalFitness = 1;
              
            ccobj.CCIndividual=   zeros(1,  ccobj.CCNumDimen);
      
           ccobj.NumSP = length(Dimen);
           ccobj.SPFitList = zeros(1, length(Dimen));
           
             ccobj.DimenIndex = Dimen; 
              total =  ccobj.DimenIndex(1);
              for i=2:ccobj.NumSP
                 total = total + Dimen(i);
                 ccobj.DimenIndex(i) = total;
              end
           
               ccobj.DimenIndex  = horzcat([0], ccobj.DimenIndex); 
           %-------------------------------------------------------
            ccobj.SP = cell(10,1); % build a cell vector of Subpopulations of type EvoAlg
            
               for n=1:ccobj.NumSP
                   ccobj.SP{n}=  ccobj@EvoAlg(PopSize,Dimen(n), Prob, ProbMax, ProbMin);
                  
               end 
               % ccobj.DimenIndex
         
           for sp =1:ccobj.NumSP 
                 ccobj.Table(sp).SolVec =   ccobj.SP{sp}.Population(1).Individual;  %Individuals for SPs - note in EvoAlg class, PopMat is used as CMAES needs Matrix to work
                 ccobj.Index(sp).IncreIndex =  ccobj.DimenIndex(1:end-(ccobj.NumSP-sp) ) ; % excludes dim of other SP
                 ccobj.Index(sp).Dim  = Dimen(1:end-(ccobj.NumSP-sp)); % expluses dim of other SP
                 ccobj.Index(sp).TaskIndividual=   zeros(1, ccobj.DimenIndex(sp+1));
                  %  ccobj.Index(sp).TaskIndividual
           end
           
             ccobj.TaskSuccess = zeros(1, ccobj.NumSP);
             
           ccobj.CCIndividual = normrnd(0, 0.01, 1,ccobj.CCNumDimen);
         
         
        end
         
      function PrintCCGA(ccobj) % print to test when needed
   
         for sp = 1:ccobj.NumSP 
           sp
               ccobj.SP{sp}.Fitness
             
           for i = 1:ccobj.SP{sp}.PopSize
             ccobj.SP{sp}.Population(i).Individual 
            end 
            ccobj.SP{sp}.N
              ccobj.SP{sp}.FinalFitness 
             ccobj.SP{sp}.FinalSolution  
          
           
         end 
      end
      
      function ccobj = GetBestTable(ccobj)
          
          for sp =1:ccobj.NumSP 
            bestIdx =  ccobj.SP{sp}.FinalFitIndex;
          
              ccobj.Table(sp).SolVec =   ccobj.SP{sp}.Population(bestIdx).Individual;  
          
            
          end 
          
      end
      
      function ccobj = Join(ccobj)
          
          BestInd=[];
           
        for sp = 1:ccobj.NumSP   
             BestInd = [BestInd,ccobj.Table(sp).SolVec] ;
        end 
      ccobj.CCIndividual = BestInd; 
      end
      
      
      
       function ccobj = JoinTask(ccobj, spindex)
          
          BestInd=[];
           
        for sp = 1:spindex   
             BestInd = [BestInd,ccobj.Table(sp).SolVec] ;
        end 
      ccobj.Index(spindex).TaskIndividual = BestInd; 
      end
      
      
      
      function ccobj =  TransferLearningtoBigNet(ccobj, Sol, SmallHid, BigHid, FitList)
          
          %SmallHid - number of hidden neurons of small network
          %BigHid
          
          % idea is to transfer the small island of network (less h) to
          % bigger island of network
%              
             index = 1; 
             
        
             
          for sp = 1:SmallHid   % copy NSP based first layer weights
               
              for ch =1:ccobj.CCDimen(sp)
                ccobj.SP{sp}.Population(1).Individual(ch) = Sol(index); 
                index = index+1 ; 
              end 
            ccobj.SP{sp}.Fitness = FitList(sp); 
              temp =   sp+1; % to continue later
          end 
            
          for sp = SmallHid + 1 + (BigHid - SmallHid): ccobj.NumSP % copy second layer (leave those that need not copy)
            
             
              for ch =1:SmallHid+1
                 ccobj.SP{sp}.Population(1).Individual(ch) = Sol(index); 
                 index = index+1  ;
              end 
            ccobj.SP{sp}.Fitness = FitList(temp); 
               
          end 
           
          
           
      end
      
       
       
  
       
      
      
      % main evolution
      % -------------------------------------------------------------------------
      function ccobj = CCEvolution(ccobj,   MaxFE,  depth, MinError,  TaskTopo,  net)
           
       %   We dont cooperatively evaluate in the begining - we leave it for
       %   evolution to evaluate 
           
        % [10, 5, 6]
       
          Cycle = 1;
          DepthSearch = depth;
        while ccobj.CCFE < MaxFE
            
             for  sp=1:ccobj.NumSP 
                 ccobj.NumSP
              %     sp
               for depth=1:DepthSearch   
                    ccobj = CooperativeCoevolution.GetBestTable(ccobj);
                 ccobj = CooperativeCoevolution.JoinTask(ccobj, sp);
                % ccobj.Index(sp).TaskIndividual
               ccobj.SP{sp} =     EvoAlg.PureCMAES(ccobj.SP{sp}, DepthSearch ,sp, ccobj.Index(sp).TaskIndividual, ccobj.Index(sp).Dim, ccobj.Index(sp).IncreIndex , TaskTopo , net  ) ;
                 ccobj.CCFE = ccobj.CCFE + ccobj.SP{sp}.PopSize * DepthSearch; 
               end 
                  ccobj.SPFitList(sp) = ccobj.SP{sp}.FinalFitness;
                %   ccobj.SPFitList(sp)
               if ccobj.SPFitList(sp)< MinError(1)
                   %  sp
                  ccobj.TaskSuccess(sp) = 1;
                      
               end
%              
              
             end
             
             
              Cycle= Cycle +1  ;
               
         
            ccobj.CCFinalFitness = ccobj.SP{ccobj.NumSP}.FinalFitness  ;
             
        end
              
            
      end
   end
end
