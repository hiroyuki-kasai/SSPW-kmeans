% Add folders to path.

addpath(pwd);

cd algorithm/;
addpath(genpath(pwd));
cd ..;

cd tools/;
addpath(genpath(pwd));
cd ..;

% if ismac
%     cd mosek/mosek_mac/;
%     addpath(genpath(pwd));
%     cd ../..;
% %elseif isunix
% %    % Code to run on Linux platform
% elseif ispc
%     cd mosek/mosek_win/;
%     addpath(genpath(pwd));
%     cd ../..;
% else
%     disp('Platform not supported')
% end





