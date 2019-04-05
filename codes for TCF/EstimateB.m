function B = EstimateB(U,V,vec,beta)

d = size(U,2);

uIDX = vec(:,1);
vIDX = vec(:,2);

U2 = U(uIDX,:);
V2 = V(vIDX,:);

Xr = U2'*(repmat(vec(:,3), [1,d]).* V2);

XX=[];
for i = 1:d
    for j=1:d
        tmp = U2' * ( repmat(U2(:,i).*V2(:,j), [1,d]) .* V2 );        
        XX(:,(j-1)*d+i) = tmp(:);
    end
end

w = ( XX + beta*eye(size(XX)) ) \ Xr(:); % sovle the linear system
B = reshape(w,d,d);
