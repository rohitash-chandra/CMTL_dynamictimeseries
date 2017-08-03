
function [Dimen, D] = SetCCNN(Topology,  H, I) 
            
             W1 = Topology(1)*Topology(2); % W1
             W2 = Topology(2)  * Topology(3); %W2
             B = Topology(2)  + Topology(3); % Bias   
            % B = Topology(2) ; % Bias   
             D = W1+W2 +B;  % total dimension  of smallest network for 1st SP
            
             Dimen(1) = D; %first SP
                
%                for n = 1: length(H)-1 %rest of SP 
%                    Hdiff = H(n+1) - H(n); 
%                     Indiff = I(n+1) - I(n);
%                     
%                     
%                     Dimen(n+1) = ((Topology(1) +1 + Topology(3)) *  Hdiff) + (Idiff *  ( Topology(2)   )) ; %input links + bias + output links 
%                end

               for n = 1: length(I)-1 %rest of SP 
                   Hdiff = H(n+1) - H(n); 
                   Idiff = I(n+1) - I(n); 
                   
                   InHid =   Idiff *  ( H(n+1)   ) 
                   HidOut =  (I(n+1) +1 + Topology(3)) *  Hdiff  
                   Dimen(n+1) = InHid  + HidOut ; %input links + bias + output links 
               end   

               
               Dimen
end
