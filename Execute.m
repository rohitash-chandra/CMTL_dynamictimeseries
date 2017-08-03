% CCGA for FNN by Rohitash Chandra, 2016: c.rohitash(at)gmail.com
%note this is for hidden layer. But many other layer can be done

% can use for classfication and also for time series predition or
% regression. 
clear all
clc

out1 = fopen('out1Final.txt', 'w');
out2 = fopen('out2Final.txt', 'w');
out3 = fopen('out3Final.txt', 'w');

% Declare NN here

            
 NumSteps =  3;


MaxFE = 30000 * NumSteps ; % 4bit  
 

 MinError = [0.001]; %Min Error for each problem
 NumProb = 7;
  ProbMin = [-5]; % initial pop range
  ProbMax = [5];


 
 Decom = 5; % type of decomposition of NN. 1 = Layer level, 2 = Network level, 3 = NSP, 4 = Synapse level, 5 is ESP

MaxRun = 30 ;

for app = 1:NumProb 
  for depth = 8:4:8
        Suc = 0; % when minimum error is satisfied
        
        SucT1 = 0;
        SucT2 = 0;
        SucT3 = 0;
        
  for Run=1:MaxRun
           
           Input = [3, 5, 7];  %  set of input size for different tasks
           
           Output = 1; 
 
           H  = [5, 7, 9];
 
           for t=1:NumSteps

            Topology{t} = [Input(t), H(t) , Output];
           Topology{t}
              
           end 
            
            H
           [Dimen, D] = SetCCNN(Topology{1},   H, Input);
            
           Dimen
          
            
         for step=1:NumSteps
                              %set data and NN
            [TrainInput{step}, TrainTarget{step}, ValidInput{step}, ValidTarget{step}, TestInput{step}, TestTarget{step}] = Data(app, Input(step));

             net{step} = FNNetwork(  TrainInput{step}, TrainTarget{step}, ValidInput{step}, ValidTarget{step} ,  Topology{step}, Decom); 
         end 
            
           
           PopSize = round((4+floor(3*log(D(end))))) % use D for larged Hidden neurons
     
          CCGA = CooperativeCoevolution(PopSize,Dimen, app, ProbMax, ProbMin);   
       
   
      
    LocalFE = 1000;
        
    phase =1 
    
    CurrentFE = 1;
    
    
    BestEr = ones(1,NumSteps) 
        
    while CurrentFE < MaxFE & phase < 300
        
        depth = 10;
        
     CCGA = CooperativeCoevolution.CCEvolution( CCGA, LocalFE * phase,  depth, MinError,  Topology,  net); % pass FNN as net
       
        TotalFE   =   CooperativeCoevolution.GetFE(CCGA)    ;
        SolOne =  CooperativeCoevolution.GetSolution(CCGA); % gives whole solution
        FitList =  CooperativeCoevolution.GetFitList(CCGA) ; 
    
     MinFit  = min(FitList);
      
     
      for step=1:NumSteps  
         T1Solution{step} = CooperativeCoevolution.SeprateTaskSolution(CCGA,step); 
      end
     
     for step=1:NumSteps 
       TotalEr(step) =   FitList(step); 
   
       if TotalEr(step) < BestEr(step)
          BestEr(step) = TotalEr(step) 
          BestSolution{step} = T1Solution{step}; 
       end
     end
     
      BestEr
      FitList
      
      
     CurrentFE = TotalFE  
     
     phase = phase +1 
     %trans
    end
     
    
  for s=1:NumSteps   
    net{s} = FNNetwork.SaveTrainedNet(net{s}, BestSolution{s},  Topology, s); 
    net{s} =  FNNetwork. TestRegressionNetwork(net{s}) ;  
    T{s}.Test(Run) = FNNetwork. GetTestRMSE(net{s}); 
     Fit{s}.Error(Run) = FNNetwork. GetTrainRMSE(net{s}); 
     
      NMSET{s}.Test(Run) = FNNetwork. GetTestNMSE(net{s}); 
     NMSEFit{s}.Error(Run) = FNNetwork. GetTrainNMSE(net{s}); 
  end
       
  
  
    fprintf(out1,'%d  %d %d \n',   app,  depth, Run); 
    
    for step=1:NumSteps  
     fprintf(out1, '%.6f  ', Fit{step}.Error(Run)); 
    end
    
      fprintf(out1,'\n'); 
      
    for step=1:NumSteps 
     fprintf(out1,'%.6f ',    T{step}.Test(Run));
    end
    
      fprintf(out1,'\n'); 
  end 
  
  
  for st=1:NumSteps 
      
      
         MeanTrain(st) = mean(Fit{st}.Error) ;
         STDTrain(st) = 1.96 *(std(Fit{st}.Error)/sqrt(MaxRun)) ;
         MeanTest(st) = mean(T{st}.Test) ;
         STDTest(st) = 1.96 *(std(T{st}.Test)/sqrt(MaxRun)) ;
  end
  
  
  for st=1:NumSteps 
      
      
         NMeanTrain(st) = mean(NMSEFit{st}.Error) ;
         NSTDTrain(st) = 1.96 *(std(NMSEFit{st}.Error)/sqrt(MaxRun)) ;
         NMeanTest(st) = mean(NMSET{st}.Test) ;
         NSTDTest(st) = 1.96 *(std(NMSET{st}.Test)/sqrt(MaxRun)) ;
  end
  
  
%      

     testmean = sum(MeanTest)/NumSteps ;
     
     trainmean = sum(MeanTrain)/NumSteps ;
%          
     teststd =    sum(STDTest)/NumSteps  ;
     
 
     trainstd =    sum(STDTrain)/NumSteps  ;
     
     fprintf(out2,'%d  %d \n',   app,  depth); 
     
     for step=1:NumSteps 
        fprintf(out2, '%.6f ',  MeanTrain(step));
     end
      fprintf(out2,'\n');  
      
     for step=1:NumSteps
        fprintf(out2, '%.6f ', STDTrain(step));
     end
     
      fprintf(out2,'\n'); 
      
     for step=1:NumSteps
     fprintf(out2, '%.6f ', MeanTest(step));
     end
     
      fprintf(out2,'\n'); 
      
     for step=1:NumSteps
     fprintf(out2, '%.6f ',  STDTest(step));
     end
     
      fprintf(out2,'\n \n'); 
     
      fprintf(out2,'%.6f %.6f %.6f %.6f \n \n \n', trainmean ,trainstd, testmean, teststd);
      
      %%%%%%%%%%%%%%%%%%
      
       testmean = sum(NMeanTest)/NumSteps ;
     
     trainmean = sum(NMeanTrain)/NumSteps ;
%          
     teststd =    sum(NSTDTest)/NumSteps  ;
     
 
     trainstd =    sum(NSTDTrain)/NumSteps  ;
     
     fprintf(out3,'%d  %d \n',   app,  depth); 
     
     for step=1:NumSteps 
        fprintf(out3, '%.6f ',  NMeanTrain(step));
     end
      fprintf(out3,'\n');  
      
     for step=1:NumSteps
        fprintf(out3, '%.6f ', NSTDTrain(step));
     end
     
      fprintf(out3,'\n'); 
      
     for step=1:NumSteps
     fprintf(out3, '%.6f ', NMeanTest(step));
     end
     
      fprintf(out3,'\n'); 
      
     for step=1:NumSteps
     fprintf(out3, '%.6f ',  NSTDTest(step));
     end
     
      fprintf(out3,'\n \n'); 
     
      fprintf(out3,'%.6f %.6f %.6f %.6f \n \n \n', trainmean ,trainstd, testmean, teststd);
      

  end
end


fclose(out1);
fclose(out2);
