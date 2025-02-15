type ('a, 'b) homotopy =
| H_id         : ('a, 'a) homotopy
| H_const      : 'b -> ('a, 'b) homotopy
| H_comp       : ('a, 'b) homotopy * ('b, 'c) homotopy -> ('a, 'c) homotopy
| H_inv        : ('a, 'b) homotopy -> ('b, 'a) homotopy
| H_pointwise  : ('a -> ('b, 'c) homotopy) * 'a -> ('a, 'c) homotopy
| H_lift       : ('a, 'b) homotopy * ('c -> 'a) -> ('c -> 'b, 'c -> 'a) homotopy
| H_path       : 'a path -> ('a, 'a) homotopy
| H_compose    : ('a -> ('b, 'c) homotopy) * ('d -> ('c, 'e) homotopy) * ('d -> ('a, 'b) homotopy)
-> ('d -> ('a, 'e)) homotopy
| H_product    : ('a -> ('b, 'c) homotopy) * ('a -> ('d, 'e) homotopy)
-> ('a, ('b, 'd), ('c, 'e)) homotopy

and 'a path = {
start  : 'a;
finish : 'a;
}
