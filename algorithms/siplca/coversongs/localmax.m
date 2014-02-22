function m = localmax(x)
% return 1 where there are local maxima in x (columnwise).
% don't include first point, maybe last point

[nr,nc] = size(x);

if nr == 1
  lx = nc;
elseif nc == 1
  lx = nr;
  x = x';
else
  lx = nr;
end

if (nr == 1) || (nc == 1)

  m = (x > [x(1),x(1:(lx-1))]) & (x >= [x(2:lx),1+x(lx)]);

  if nc == 1
    % retranspose
    m = m';
  end
  
else
  % matrix
  lx = nr;
  m = (x > [x(1,:);x(1:(lx-1),:)]) & (x >= [x(2:lx,:);1+x(lx,:)]);

end
