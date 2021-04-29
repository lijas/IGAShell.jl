
struct MyLagrange{dim,shape,order} <: Interpolation{dim,shape,order} end

function generate_1d_lagrange_value(order::Int, i::Int, ξ::Vec{1})

    @assert(i<=(order+1))
    x = range(-1.0, stop=1.0, length=order+1)

    _val = 1.0
    for m in 1:(order+1)
        m==i && continue
        t = ξ[1] - x[m]
        n = x[i] - x[m]
        _val *= (t/n)
    end
    return _val
end

function Ferrite.value(ip::MyLagrange{dim,RefCube,order}, _i::Int, ξ::Vec{dim}) where {dim,order}

    orders = ntuple(_-> order, dim)
    i = _lagrange_ordering(ip)[_i]
    #@warn("Lagrange order")
    i_dim = Tuple(CartesianIndices(orders.+1)[i])
    _val = 1.0
    for i in 1:dim
        _val *= generate_1d_lagrange_value(order, i_dim[i], Vec(ξ[i]) )
    end
    return _val
end

Ferrite.getnbasefunctions(::MyLagrange{dim,RefCube,order}) where {dim,order} = (order+1)^dim
Ferrite.nvertexdofs(::MyLagrange{dim,RefCube,order}) where {dim,order} = 1
Ferrite.faces(::MyLagrange{dim,RefCube,order}) where {dim,order} = error("Not implemented")

function _lagrange_ordering(::MyLagrange{1,RefCube,order}) where {order}
    dim = 1
    orders = ntuple(_-> order, dim)

    ci = CartesianIndices((orders.+1))
    ind = reshape(1:(order+1)^dim, (orders.+1)...)

    ordering = Int[]
    
    #Corners
    corner1 = ci[1]; corner2 = ci[end]
    push!(ordering, ind[corner1])
    push!(ordering, ind[corner2])
    
    #inner dofs
    rest = ci[2:end-1]
    append!(ordering, ind[rest])

    return ordering
end

function _lagrange_ordering(::MyLagrange{2,RefCube,order}) where {order}
    dim = 2
    orders = ntuple(_-> order, dim)

    ci = CartesianIndices((orders.+1))
    ind = reshape(1:(order+1)^dim, (orders.+1)...)

    #Corners
    ordering = Int[]
    corner = ci[1,1]
    push!(ordering, ind[corner])

    corner = ci[end,1]
    push!(ordering, ind[corner])

    corner = ci[end,end]
    push!(ordering, ind[corner])

    corner = ci[1,end]
    push!(ordering, ind[corner])

    #edges
    edge = ci[2:end-1,1]
    append!(ordering, ind[edge])
    
    edge = ci[end,2:end-1]
    append!(ordering, ind[edge])

    edge = ci[2:end-1,end]
    append!(ordering, ind[edge])

    edge = ci[1,2:end-1]
    append!(ordering, ind[edge])

    #inner dofs, ordering??
    rest = ci[2:end-1,2:end-1]
    append!(ordering, ind[rest])
    return ordering
end
#Numbering:
#https://blog.kitware.com/wp-content/uploads/2020/03/Implementation-of-rational-Be%CC%81zier-cells-into-VTK-Report.pdf
function _lagrange_ordering(::MyLagrange{3,RefCube,order}) where {order}
    dim = 3
    orders = ntuple(_-> order, dim)

    ci = CartesianIndices((orders.+1))
    ind = reshape(1:(order+1)^dim, (orders.+1)...)

    #Corners, bottom
    ordering = Int[]
    corner = ci[1,1,1]
    push!(ordering, ind[corner])

    corner = ci[end,1,1]
    push!(ordering, ind[corner])

    corner = ci[end,end,1]
    push!(ordering, ind[corner])

    corner = ci[1,end,1]
    push!(ordering, ind[corner])

    #Corners, top
    corner = ci[1,1,end]
    push!(ordering, ind[corner])

    corner = ci[end,1,end]
    push!(ordering, ind[corner])

    corner = ci[end,end,end]
    push!(ordering, ind[corner])

    corner = ci[1,end,end]
    push!(ordering, ind[corner])

    #edges, bottom
    edge = ci[2:end-1,1,1]
    append!(ordering, ind[edge])
    
    edge = (ci[end,2:end-1,1])
    append!(ordering, ind[edge])

    edge = (ci[2:end-1,end,1]) # Reverse?
    append!(ordering, ind[edge])

    edge = (ci[1,2:end-1,1]) # Reverse?
    append!(ordering, ind[edge])

    #edges, top
    edge = ci[2:end-1,1,end]
    append!(ordering, ind[edge])
    
    edge = (ci[end,2:end-1,end])
    append!(ordering, ind[edge])

    edge = (ci[2:end-1,end,end]) # Reverse?
    append!(ordering, ind[edge])

    edge = (ci[1,2:end-1,end]) # Reverse?
    append!(ordering, ind[edge])

    #edges, mid
    edge = (ci[1,1,2:end-1])
    append!(ordering, ind[edge])
    
    edge = (ci[end,1,2:end-1])
    append!(ordering, ind[edge])

    edge = (ci[end,end,2:end-1]) # Reverse?
    append!(ordering, ind[edge])

    edge = (ci[1,end,2:end-1]) # Reverse?
    append!(ordering, ind[edge])

    #Faces (vtk orders left face first, but Ferrite orders bottom first)
    #Face, bottom
    face = ci[2:end-1,2:end-1,1][:] #bottom
    append!(ordering, ind[face])
    
    face = ci[2:end-1,1,2:end-1][:] #front
    append!(ordering, ind[face])

    face = ci[end,2:end-1,2:end-1][:] #left
    append!(ordering, ind[face])

    face = ci[2:end-1,end,2:end-1,end][:] #back
    append!(ordering, ind[face])

    face = ci[1, 2:end-1,2:end-1,end][:] #right
    append!(ordering, ind[face])

    face = ci[2:end-1,end,2:end-1,end][:] #top
    append!(ordering, ind[face])

    #

    #inner dofs, ordering??
    rest = ci[2:end-1,2:end-1,2:end-1][:]
    append!(ordering, ind[rest])
    return ordering
end

function tes_mylagrange()

    dim = 2
    qr = QuadratureRule{dim,RefCube}(2)
    for i in 1:length(qr.weights)
        for order in [1,2]
            ordering = _lagrange_ordering(MyLagrange{dim,RefCube,order}())
            for j in 1:getnbasefunctions(Lagrange{dim,RefCube,order}())
                val1 = Ferrite.value(Lagrange{dim,RefCube,order}(), j, qr.points[i])
                val2 = value(MyLagrange{dim,RefCube,order}(), ordering[j], qr.points[i])
                @show j, val1 ≈ val2
            end
        end
    end
end