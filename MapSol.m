
 function [TransSol] = MapSol(Sol1, Sol2, H1, H2,   Tp) 
%function [TransSol] = MapSol()   
 
 % Sol1 = [1, 3, 2, 4, 5, 6]; %no bias
  %Sol2 = [1, 3, 2, 4, 5, 6, 7, 8, 9];
% 
% S = [1, 3, 2, 4, 7, 8, 5, 6, 9];
%  Tp = [2,2,1];

 %Sol1 = [1, 3, -1,  2, 4, -2,  5, 6, -3] 
 % Sol2 = [1, 3, -1,  2, 4, -2,  5, 6, -3, 7, 8, -4, 9] 
 
 
  %Sol1 = [1, 3, -1,  2, 4, -2,  5, 6, -3,  30, 31, -10]  %not working
   %Sol2 = [1, 3, -1,  2, 4, -2,  5, 6, -3, 7, 8, -4, 9, 10] 
% 
% S = [1, 3, 2, 4, 7, 8, 5, 6, 9];
  %Tp = [2,2,1];
             
  S1W1 = (Tp(1) * H1)+H1; 
  S1W2 = (H1 * Tp(3))+Tp(3);
 
 S2W1 = (Tp(1) * H2)+H2; 
  S2W2 = (Tp(3) * H2)+Tp(3);
 
   %S1W1 = (Tp(1) * H1) ; 
 %S1W2 = (H1 * Tp(3))+Tp(3);
 
 %%S2W1 = (Tp(1) * H2) ; 
 %S2W2 = (Tp(3) * H2)+Tp(3);
 
             InHidS1 = Sol1(1: S1W1  ) % all input to hid weights in SP1
             HidOutS1 = Sol1( S1W1+1 :end)
             
             InHidS2 = Sol2(1: S2W1  ) %   SP1
           HidOutS2 = Sol2( S2W1+1 :end -((H2-H1)*Tp(3) )) % SP2
          %  HidOutS2 = Sol2( S2W1+1 :end ) % SP2
             X = InHidS2(S1W1+1:end - Tp(3)) % whatever is left at end of InHidS2 
             BOut = InHidS2(end - Tp(3) +1:end );
              
             WOut = Sol2(end - (  Tp(3)*(H2-H1))+1:end)   % weight links from added H to Output
             
                   TransSol =  horzcat(InHidS1, HidOutS2,X, WOut, BOut)
             
              
end
