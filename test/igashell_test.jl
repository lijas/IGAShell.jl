function calculate_element_volume(cell_values::eliasfem.IGAShellValues, nlayers::Int)
    qp = 0
    V = 0
    for ilay in 1:nlayers
        eliasfem.reinit_layer!(cell_values, ilay)
        for qp in 1:eliasfem.getnquadpoints_per_layer(cell_values)
            V += getdetJdV(cell_values, qp)
        end
    end
    return V
end

function calculate_element_area(cell_values::eliasfem.IGAShellValues, INDEX, nlayers::Int)
    A = 0
    for ilay in 1:nlayers
        eliasfem.reinit_layer!(cell_values, ilay)
        for qp in 1:eliasfem.getnquadpoints_per_layer(cell_values)
            A += eliasfem.getdetJdA(cell_values, qp, INDEX)
        end
    end
    return A
end

function get_and_reinit_cv(igashell, grid, cellid)
    cv = eliasfem.build_cellvalue!(igashell, cellid)
    Ce = eliasfem.get_extraction_operator(eliasfem.intdata(igashell), cellid)
    IGA.set_bezier_operator!(cv, Ce)
    
    coords = getcoordinates(grid, cellid)
    bezier_coords = IGA.compute_bezier_points(Ce, coords)
    
    eliasfem.reinit_midsurface!(cv, bezier_coords)
    return cv
end

function get_and_reinit_fv(igashell, grid, index)
    cv = eliasfem.build_facevalue!(igashell, index)
    Ce = eliasfem.get_extraction_operator(eliasfem.intdata(igashell), index[1])
    IGA.set_bezier_operator!(cv, Ce)

    coords = getcoordinates(grid, index[1])
    bezier_coords = IGA.compute_bezier_points(Ce, coords)
    
    eliasfem.reinit_midsurface!(cv, bezier_coords)
    return cv
end

function get_curved_mesh(cellstate; h, b, R)

    dim = 3; T = Float64
    ν = 0.4; E = 1e5
    visc_para = 0.0
    
    #Mesh
    orders = (3,3); r = 3
    angles = deg2rad.(T[0,90,0])
    nlayers = length(angles)
    ninterfaces = nlayers-1
    
    nelx = 100; nely = 1
    nurbsmesh = eliasfem.IGA.generate_curved_nurbsmesh((nelx,nely), orders, pi/2, R, b, multiplicity=(1,1))
    grid = eliasfem.IGA.convert_to_grid_representation(nurbsmesh)
    
    cellstates = [cellstate for i in 1:nelx*nely]

    interface_damage = [0.0 for _ in 1:ninterfaces, _ in 1:nelx*nely]
    
    #Material
    interfacematerial = eliasfem.MatCohesive{dim,T}(-1.0, -1.0, -1.0, 1.0, 1.0)
    layermats = [MatLinearElastic{dim}(1.0, E, ν) for i in 1:nlayers]
    layer_mats = eliasfem.LayeredMaterial(layermats, angles)
    
    
    #IGAshell
    #igashelldata = eliasfem.IGAShellData{dim}(layer_mats, interfacematerial, visc_para, (orders...,r), nurbsmesh.knot_vectors, h, dim==2 ? b : 1.0, nlayers, cellstates, interface_damage, adaptive, linear, 4, 3, 4)
    igashelldata = 
    eliasfem.IGAShellData(;
        layer_materials           = layer_mats,
        interface_material        = interfacematerial,
        viscocity_parameter       = 0.0,
        orders                    = (orders..., r),
        knot_vectors              = nurbsmesh.knot_vectors,
        thickness                 = h,
        initial_cellstates        = cellstates,
        #initial_interface_damages          = interface_damage,
        width                     = dim == 2 ? b : 1.0,
        adaptable                 = false,
        small_deformations_theory = true,
        nqp_inplane_order         = 4,
        nqp_ooplane_per_layer     = 4,
        nqp_interface_order       = 2,
    ) 
    igashell = eliasfem.IGAShell(collect(1:getncells(grid)), reverse(nurbsmesh.IEN, dims=1), igashelldata) 

    return grid, igashell
end

function get_cube_mesh(cellstate; h, b, L)

    dim = 3; T = Float64
    
    #Mesh
    orders = (3,3); r = 3 
    angles = deg2rad.(T[0,90,0])
    nlayers = length(angles)
    ninterfaces = nlayers-1
    
    nelx = 10; nely = 1
    nurbsmesh = eliasfem.IGA.generate_nurbsmesh((nelx,nely), orders, (L,b), sdim=dim)
    grid = eliasfem.IGA.convert_to_grid_representation(nurbsmesh)
    
    cellstates = [cellstate for i in 1:nelx*nely]
    
    #Material
    interfacematerial = eliasfem.MatCohesive{dim,T}(-1.0, -1.0, -1.0, 1.0, 1.0)
    layermats = [MatLinearElastic{dim}(1.0, 1e5, 0.4) for i in 1:nlayers]
    layer_mats = eliasfem.LayeredMaterial(layermats, angles)
    
    
    #IGAshell
    igashelldata = 
    eliasfem.IGAShellData(;
        layer_materials           = layer_mats,
        interface_material        = interfacematerial,
        orders                    = (orders..., r),
        knot_vectors              = nurbsmesh.knot_vectors,
        thickness                 = h,
        initial_cellstates        = cellstates,
        width                     = dim == 2 ? b : 1.0,
        adaptable                 = false,
        small_deformations_theory = true,
        nqp_inplane_order         = 4,
        nqp_ooplane_per_layer     = 2,
        nqp_interface_order       = 4,
    ) 
    igashell = eliasfem.IGAShell(collect(1:getncells(grid)), reverse(nurbsmesh.IEN, dims=1), igashelldata) 

    return grid, igashell
end

@testset "igashellvalues_curved" begin

    # # #
    # TEST IGASHELL VALUES
    # # #

    h = 0.2;    b = .556;     R = 4.22;
    grid, igashell = get_curved_mesh(eliasfem.LAYERED, h=h, b=b, R=R)
    addedgeset!(grid, "left", (x)-> isapprox(x[1], 0.0, atol=1e-5))
    addedgeset!(grid, "right", (x)-> x[3]≈0.0)
    addedgeset!(grid, "front", (x)-> x[2]≈0.0)
    
    rightface = collect(getedgeset(grid, "right"))
    rightedge = [eliasfem.EdgeInterfaceIndex(edgeidx..., 1) for edgeidx in rightface]
    leftface  = collect(getedgeset(grid, "left"))
    leftedge = [eliasfem.EdgeInterfaceIndex(edgeidx..., 2) for edgeidx in leftface]
    frontface  = collect(getedgeset(grid, "front"))
    frontedgetop = [eliasfem.EdgeInterfaceIndex(edgeidx..., 2) for edgeidx in frontface]
    frontedgebot = [eliasfem.EdgeInterfaceIndex(edgeidx..., 1) for edgeidx in frontface]

    #Volume
    V = 0.0
    for cellid in 1:getncells(grid)

        cv = get_and_reinit_cv(igashell, grid, cellid)

        V += calculate_element_volume(cv, eliasfem.nlayers(igashell))
    end
    @test isapprox(V, pi/2 * R * b * h, atol=1e-3)

    #Edgelength 1
    L = 0.0
    for edge in rightedge
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge, eliasfem.nlayers(igashell))
    end
    @test L ≈ b

    #Edgelength 2
    L = 0.0
    for edge in leftedge
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge, eliasfem.nlayers(igashell))
    end
    @test L ≈ b

    #Edgelength 3
    L = 0.0
    for edge in frontedgetop
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge, eliasfem.nlayers(igashell))
    end
    @test isapprox(L, (R+h/2)*pi/2, atol=1e-3)
    
    #Edgelength 4

    L = 0.0
    for edge in frontedgebot
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge, eliasfem.nlayers(igashell))
    end
    @test isapprox(L, (R-h/2)*pi/2, atol=1e-3)

    #Side area 1
    A = 0.0
    for edge in rightface
        cellid, edgeid = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        A  += calculate_element_area(cv,edge, eliasfem.nlayers(igashell))
    end
    @test A ≈ h*b

    #Side area 2
    A = 0.0
    for edge in leftface
        cellid, edgeid = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        A  += calculate_element_area(cv, edge, eliasfem.nlayers(igashell))
    end
    @test A ≈ h*b

    #Side area 3
    A = 0.0
    for edge in frontface
        cellid, edgeid = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        A  += calculate_element_area(cv, edge, eliasfem.nlayers(igashell))
    end
    @test isapprox(A, R*pi/2 * h, atol=1e-3)

    #Check curvature
    cv = get_and_reinit_cv(igashell, grid, getncells(grid)÷2)
    κ = getindex.(cv.κᵐ,1,1)
    @test all( isapprox.(κ, 1/R, atol=1e-2) )


    # # #
    # TEST LOCAL DOF GETTER
    # # #

    @test eliasfem.igashelldofs(igashell, first(frontedgebot)) == [1, 2, 3, 31, 32, 33, 61, 62, 63, 91, 92, 93]

end

@testset "igashell utils" begin

    order = 2;
    ninterfaces = 1
    @test (eliasfem.generate_knot_vector(order, ninterfaces, 0) .≈ Float64[-1,-1,-1, 1,1,1]) |> all
    @test (eliasfem.generate_knot_vector(order, ninterfaces, 3) .≈ Float64[-1,-1,-1, 0,0,0, 1,1,1]) |> all
    @test (eliasfem.generate_knot_vector(order, ninterfaces, 4) .≈ Float64[-1,-1,-1, 0,0,0,0, 1,1,1]) |> all

    ninterfaces = 2
    @test (eliasfem.generate_knot_vector(order, ninterfaces, 1) .≈ Float64[-1,-1,-1, -1/3, 1/3, 1,1,1]) |> all

    ninterfaces = 2
    @test (eliasfem.generate_knot_vector(order, ninterfaces, [1,2]) .≈ Float64[-1,-1,-1, -1/3, 1/3,1/3, 1,1,1]) |> all
    @test (eliasfem.generate_knot_vector(order, ninterfaces, [2,1]) .≈ Float64[-1,-1,-1, -1/3, -1/3,1/3, 1,1,1]) |> all

    ninterfaces = 3
    @test (eliasfem.generate_knot_vector(order, ninterfaces, [1,1,2]) .≈ Float64[-1,-1,-1, -0.5, 0.0, 0.5,0.5, 1,1,1]) |> all

    order = 1; ninterfaces = 3
    @test (eliasfem.generate_knot_vector(eliasfem.LUMPED, order, ninterfaces) .≈ Float64[-1,-1, 1,1]) |> all
    @test (eliasfem.generate_knot_vector(eliasfem.LAYERED, order, ninterfaces) .≈ Float64[-1,-1, -5/10, 0, 5/10, 1,1]) |> all
    @test (eliasfem.generate_knot_vector(eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(1), order, ninterfaces) .≈ Float64[-1,-1, -5/10, -5/10, 1,1]) |> all
    @test (eliasfem.generate_knot_vector(eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(3), order, ninterfaces) .≈ Float64[-1,-1, 5/10, 5/10, 1,1]) |> all
    @test (eliasfem.generate_knot_vector(eliasfem.STRONG_DISCONTINIUOS_AT_INTERFACE(2), order, ninterfaces) .≈ Float64[-1,-1, -0.5, 0.0,0.0, 0.5, 1,1]) |> all

    ##
    dim_s = 3;
    ooplane_order = 2; 
    nlayers = 2; ninterfaces = nlayers-1;
    @test 9 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.LUMPED)
    @test 15 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.LAYERED)
    @test 18 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.STRONG_DISCONTINIUOS(1))
    @test 18 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.WEAK_DISCONTINIUOS(1))

    ##
    dim_s = 3;
    ooplane_order = 1; 
    nlayers = 4; ninterfaces = nlayers-1;
    @test  6 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.LUMPED)
    @test 15 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.LAYERED)
    @test 18 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.STRONG_DISCONTINIUOS(1))
    @test 18 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.STRONG_DISCONTINIUOS(2))
    @test 21 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.STRONG_DISCONTINIUOS(3))
    @test 18 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.STRONG_DISCONTINIUOS(4))
    @test 12 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(3))
    @test 12 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(2))
    @test 18 == eliasfem.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, eliasfem.WEAK_DISCONTINIUOS(3))

    ## Bezier extraction matrix
    nlayers = 3; order = 1
    knot_lumped = eliasfem.generate_knot_vector(order, nlayers-1, 0)

    #lumped to layered
    new_knots = [-1 + 2i/(nlayers) for i in 1:(nlayers-1)]
    Cmat_lu2la = eliasfem.generate_out_of_plane_extraction_operators(knot_lumped, order, new_knots, fill(order, nlayers-1))

    #lumped to discont
    knot_layered = eliasfem.generate_knot_vector(order, nlayers-1, order)
    Cmat_la2di = eliasfem.generate_out_of_plane_extraction_operators(knot_layered, order, new_knots, [1,0])
    @test all(Cmat_la2di*Cmat_lu2la .≈ [1.0 0.0; 2/3 1/3; 2/3 1/3; 1/3 2/3; 0.0 1.0])

    #Multiplicity vector
    ninterfaces = 4; order = 2
    @test ([3,0,0,0] .== eliasfem.generate_nmultiplicity_vector(eliasfem.WEAK_DISCONTINIUOS(1), ninterfaces, order)) |> all
    @test ([3,2,2,2] .== eliasfem.generate_nmultiplicity_vector(eliasfem.STRONG_DISCONTINIUOS(1), ninterfaces, order)) |> all

    ## Adaptivity stuff
    h = 0.2;    b = .556;     R = 4.22;
    grid, igashell = get_curved_mesh(eliasfem.LAYERED, h=h, b=b, R=R)

    eliasfem.get_upgrade_operator(eliasfem.adapdata(igashell), eliasfem.LUMPED, eliasfem.LAYERED)
    eliasfem.get_upgrade_operator(eliasfem.adapdata(igashell), eliasfem.WEAK_DISCONTINIUOS(1), eliasfem.FULLY_DISCONTINIUOS)
    eliasfem.get_upgrade_operator(eliasfem.adapdata(igashell), eliasfem.STRONG_DISCONTINIUOS(1), eliasfem.FULLY_DISCONTINIUOS)
    eliasfem.get_upgrade_operator(eliasfem.adapdata(igashell), eliasfem.LUMPED, eliasfem.WEAK_DISCONTINIUOS(1))
    #eliasfem.@showm eliasfem.IGA.beo2matrix(C)'

    #Active basefunctions in layer
    order = 2; ilay = 1; ninterfaces=2
    @test (eliasfem.get_active_basefunctions_in_layer(1, order, eliasfem.LUMPED) .== 1:order+1) |> all
    @test (eliasfem.get_active_basefunctions_in_layer(2, order, eliasfem.LUMPED) .== 1:order+1) |> all

    @test (eliasfem.get_active_basefunctions_in_layer(1, order, eliasfem.LAYERED) .== 1:order+1)  |> all
    @test (eliasfem.get_active_basefunctions_in_layer(2, order, eliasfem.LAYERED) .== (1:order+1) .+ order) |> all

    @test (eliasfem.get_active_basefunctions_in_layer(1, order, eliasfem.FULLY_DISCONTINIUOS) .== (1:order+1)) |> all
    @test (eliasfem.get_active_basefunctions_in_layer(2, order, eliasfem.FULLY_DISCONTINIUOS) .== (1:order+1) .+ (order+1)) |> all

    @test (eliasfem.get_active_basefunctions_in_layer(2, order, eliasfem.STRONG_DISCONTINIUOS_AT_INTERFACE(2)) .== (1:order+1) .+ (order)) |> all
    @test (eliasfem.get_active_basefunctions_in_layer(3, order, eliasfem.STRONG_DISCONTINIUOS_AT_INTERFACE(2)) .== (1:order+1) .+ 5) |> all

    @test (eliasfem.get_active_basefunctions_in_layer(1, order, eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(1)) .== (1:order+1)) |> all
    @test (eliasfem.get_active_basefunctions_in_layer(2, order, eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(1)) .== (1:order+1) .+ order.+1) |> all
    @test (eliasfem.get_active_basefunctions_in_layer(3, order, eliasfem.WEAK_DISCONTINIUOS_AT_INTERFACE(1)) .== (1:order+1) .+ order.+1) |> all

end

@testset "igashell values" begin 

#    grid, igashell = get_cube_mesh(eliasfem.LUMPED, h=0.1, b=1.0, L = 10.0)
    h = 0.2
    grid, igashell = get_curved_mesh(eliasfem.LAYERED, h=h, b=.556, R=4.22)
    dim = JuAFEM.getdim(igashell)

    for cellid in 1:1#getncells(igashell)
        cellstate = eliasfem.getcellstate(eliasfem.adapdata(igashell), cellid)

        cv = eliasfem.build_cellvalue!(igashell, cellid)
        Ce = eliasfem.get_extraction_operator(eliasfem.intdata(igashell), cellid)
        active_layerdofs = eliasfem.build_active_layer_dofs(igashell, cellstate)
        ndofs_per_cell = maximum(maximum.(active_layerdofs))

        eliasfem.IGA.set_bezier_operator!(cv, Ce)

        coords = getcoordinates(grid, cellid)
        bezier_coords = IGA.compute_bezier_points(Ce, coords)
        
        eliasfem.reinit_midsurface!(cv, bezier_coords)
        
        ue = rand(Float64, ndofs_per_cell)*0.001
        
        for ilay in 1:eliasfem.nlayers(igashell)
            eliasfem.reinit_layer!(cv, ilay)
            ue_layer = ue[active_layerdofs[ilay]]
            for qp in 1:eliasfem.getnquadpoints_per_layer(cv)
                #dU
                dUdξ = gradient( (ξ) -> eliasfem._calculate_u((cv.inp_ip, cv.oop_ip, Ce), ue, ξ), cv.qr[ilay].points[qp])
                for d in 1:dim
                    dU1 = eliasfem.function_parent_derivative(cv, qp, ue_layer, d)
                    dU2 = dUdξ[:,d]
                    @test dU1 ≈ dU2
                end
                #G
                for d in 1:dim
                    G = eliasfem._calculate_G(cv.inp_ip, bezier_coords, h, cv.qr[ilay].points[qp], d)
                    @test cv.G[qp][d] ≈  G
                end
                #g
                for d in 1:dim
                    g1 = eliasfem._calculate_g((cv.inp_ip, cv.oop_ip, Ce), bezier_coords, h, ue, cv.qr[ilay].points[qp], d)
                    g2 = cv.G[qp][d] + eliasfem.function_parent_derivative(cv, qp, ue_layer, d)
                    @test g1 ≈ g2
                end

                for i in 1:eliasfem.getnbasefunctions_per_layer(cv)
                    ξ = cv.qr[ilay].points[qp]

                    N(_ξ) = eliasfem._calculate_basefunc((cv.inp_ip, cv.oop_ip, Ce), active_layerdofs[ilay][i], _ξ)
                    @test N(ξ) ≈ cv.N[i, qp]

                    dNdξ = gradient( (_ξ) -> N(_ξ), cv.qr[ilay].points[qp])
                    @test dNdξ ≈ cv.dNdξ[i, qp]
                end

            end
        end
    end


end