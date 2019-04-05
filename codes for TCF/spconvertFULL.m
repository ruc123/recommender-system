function R = spconvertFULL(vec, num_user, num_item)
% vec: (user, item, rating)
% R: num_user * num_item

% ---
R = spconvert(vec);
[n m] = size(R);

% --- concatenate columns
if m < num_item
    R = [R, zeros(n, num_item-m)];
end
% --- concatenate rows
if n < num_user;
    R = [R; zeros(num_user-n, num_item)];
end
