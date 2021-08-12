

function NURBSValue(ip::BSplineBasis, w::Vector{Float64}, i::Int, ξ::Vec{dim,T}) where {dim,T}
    N = Ferrite.value(basis, ξ)
    W = sum(N.*w)
    R = (N[i]*w[i])./W
    return R
end

function Tensors._extract_gradient(v::Vec{3, <: Tensors.Dual}, ::Vec{2})
    p1, p2, p3 = Tensors.partials(v[1]), Tensors.partials(v[2]), Tensors.partials(v[3])
    v1 = Vec{3}((p1[1],p2[1],p3[1]))
    v2 = Vec{3}((p1[2],p2[2],p3[2]))
    return v1,v2
end

function Tensors._extract_gradient((v1,v2)::Tuple{Vec{3, <: Tensors.Dual}, Vec{3, <: Tensors.Dual}}, V::Vec{2})
    v11,v12 = Tensors._extract_gradient(v1, V)
    v21,v22 = Tensors._extract_gradient(v2, V)
    return (v11, v12, v21, v22)
end


function nurbs_shape_value(basis::BSplineBasis, ξ::Vec{pdim,T2}, (x,w)::Tuple{Vector{Vec{sdim,T}}, Vector{T}}) where {pdim,sdim,T,T2}
    order = (2,2)
    ordering = IGA._bernstein_ordering(BernsteinBasis{pdim,order}())
    N = Ferrite.value(basis, ξ)[ordering]
    W = sum(N.*w)
    R = (N.*w)./W


    pos = sum(R.*x)

    return pos
end

@testset "Test igashellvalues NURBS" begin
    
    R1 = 25.0
    R2 = 10.0
    α2 = deg2rad(95) 
    nurbsmesh = IGA.generate_nurbs_patch(:doubly_curved_nurbs, (1,1), r1 = R1, r2 = R2, α2 = α2)
    #nurbsmesh = IGA.generate_nurbs_patch(:cube, (1,1), (2,2), size = (1.0,1.0), sdim=3)
    grid = IGA.BezierGrid(nurbsmesh)

    iqr = QuadratureRule{2,RefCube}(3)
    oqr = QuadratureRule{1,RefCube}(1)
    mid_ip = BernsteinBasis{2,(2,2)}()
    oop_ip = BernsteinBasis{1,(2,)}()

    cv = IgAShell.IGAShellValues(1.0, iqr, oqr, mid_ip, 1, [oop_ip for i in 1:getnbasefunctions(mid_ip)])


    C = IGA.get_extraction_operator(grid, 1)
    xb = get_bezier_coordinates(grid, 1)
    
    x = getcoordinates(grid,1)
    w = getweights(grid, 1)

    IGA.set_bezier_operator!(cv,C,(x,w))
    IgAShell.reinit!(cv, xb)


    κ_mid = cv.κᵐ[5]
    @test (κ_mid .≈ [0.04 0.0; 0.0 0.1]) |> all

    #Check nurbs derivatives
    # Use 2x2 elements so the bezier extraction matrix is not diagonal.
    #nurbsmesh = IGA.generate_nurbs_patch(:doubly_curved_nurbs, (1,1), r1 = R1, r2 = R2, α2 = α2)

    basis = IGA.BSplineBasis(nurbsmesh.knot_vectors, nurbsmesh.orders)
    for iqp in 1:length(iqr.weights)
        ξ = iqr.points[iqp]
        (dXdξ, dXdη), X = gradient(ξ -> nurbs_shape_value(basis, ξ, (x,w)), ξ, :all)
        (d²Xdξ²,d²Xdξη,d²Xdηξ,d²Xdη²), (dXdξ, dXdη), X = hessian(ξ -> nurbs_shape_value(basis, ξ, (x,w)), ξ, :all)
        @test X ≈ shape_value(cv, iqp, xb) atol = 1e-14
        @test dXdξ ≈ IgAShell.shape_parent_derivative(cv, iqp, xb, 1) atol = 1e-14
        @test dXdη ≈ IgAShell.shape_parent_derivative(cv, iqp, xb, 2) atol = 1e-14
        @test d²Xdξ² ≈ IgAShell.shape_parent_second_derivative(cv, iqp, xb, (1,1)) atol = 1e-14
        @test d²Xdξη ≈ IgAShell.shape_parent_second_derivative(cv, iqp, xb, (1,2)) atol = 1e-14
        @test d²Xdηξ ≈ IgAShell.shape_parent_second_derivative(cv, iqp, xb, (2,1)) atol = 1e-14
        @test d²Xdη² ≈ IgAShell.shape_parent_second_derivative(cv, iqp, xb, (2,2)) atol = 1e-14
    end
end

