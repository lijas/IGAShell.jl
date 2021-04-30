function calculate_element_volume(cv::IgAShell.IGAShellValues)
    V = 0

    for qp in 1:IgAShell.getnquadpoints(cv)
        V += getdetJdV(cv, qp)
    end

    return V
end

function calculate_element_area(cv::IgAShell.IGAShellValues, INDEX)
    A = 0

    for qp in 1:IgAShell.getnquadpoints(cv)
        A += IgAShell.getdetJdA(cv, qp, INDEX)
    end

    return A
end

function get_and_reinit_cv(igashell, grid, cellid)
    cellstate = IgAShell.getcellstate(igashell, cellid)
    cv = IgAShell.build_cellvalue!(igashell, cellstate)
    Ce = IgAShell.get_extraction_operator(IgAShell.intdata(igashell), cellid)
    IgAShell.IGA.set_bezier_operator!(cv, Ce)
    
    coords = getcoordinates(grid, cellid)
    bezier_coords = IgAShell.IGA.compute_bezier_points(Ce, coords)
    
    IgAShell.reinit!(cv, bezier_coords)
    return cv
end

function get_and_reinit_fv(igashell, grid, index)

    cv = IgAShell.build_facevalue!(igashell, index)
    Ce = IgAShell.get_extraction_operator(IgAShell.intdata(igashell), index[1])
    IgAShell.IGA.set_bezier_operator!(cv, Ce)

    coords = getcoordinates(grid, index[1])
    bezier_coords = IgAShell.IGA.compute_bezier_points(Ce, coords)
    
    IgAShell.reinit!(cv, bezier_coords)
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
    #nurbsmesh = IgAShell.IgAShell.IGA.generate_curved_nurbsmesh((nelx,nely), orders, pi/2, R, b, multiplicity=(1,1))
    nurbsmesh = IgAShell.IGA.generate_nurbs_patch(:singly_curved_shell, (nelx,nely), orders; α=pi/2, R=R, width=b) 
    grid = Grid(nurbsmesh)

    cellstates = [cellstate for i in 1:nelx*nely]

    interface_damage = [0.0 for _ in 1:ninterfaces, _ in 1:nelx*nely]
    
    #Material
    interfacematerial = Five.MatCZKolluri(
        σₘₐₓ = 60 * 0.5,
        τₘₐₓ = 90 * 0.5,
        Φₙ = 211.0/1000,
        Φₜ = 1050.0/1000
    )
    layermats = [MatLinearElastic(E=E, nu=ν) for i in 1:nlayers]
    
    #IGAshell
    #igashelldata = IgAShell.IGAShellData{dim}(layer_mats, interfacematerial, visc_para, (orders...,r), nurbsmesh.knot_vectors, h, dim==2 ? b : 1.0, nlayers, cellstates, interface_damage, adaptive, linear, 4, 3, 4)
    igashelldata = 
    IgAShell.IGAShellData(;
        layer_materials           = layermats,
        interface_material        = interfacematerial,
        orders                    = (orders..., r),
        knot_vectors              = nurbsmesh.knot_vectors,
        thickness                 = h,
        initial_cellstates        = cellstates,
        #initial_interface_damages          = interface_damage,
        width                     = dim == 2 ? b : 1.0,
        adaptable                 = false,
            limit_stress_criterion   = 100.993,
            limit_damage_criterion   = 0.01,
            search_radius            = 10.0,
            locked_elements          = Int[],
        small_deformations_theory = true,
        nqp_inplane_order         = 4,
        nqp_ooplane_per_layer     = 4,
        nqp_interface_order       = 2,
    ) 

    igashell = IgAShell.IGAShell(
        cellset = collect(1:getncells(grid)), 
        data = igashelldata) 

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
    nurbsmesh = IgAShell.IgAShell.IGA.generate_nurbsmesh((nelx,nely), orders, (L,b), sdim=dim)
    grid = IgAShell.IgAShell.IGA.convert_to_grid_representation(nurbsmesh)
    
    cellstates = [cellstate for i in 1:nelx*nely]
    
    #Material
    interfacematerial = IgAShell.MatCohesive{dim,T}(-1.0, -1.0, -1.0, 1.0, 1.0)
    layermats = [MatLinearElastic{dim}(1.0, 1e5, 0.4) for i in 1:nlayers]
    layer_mats = IgAShell.LayeredMaterial(layermats, angles)
    
    
    #IGAshell
    igashelldata = 
    IgAShell.IGAShellData(;
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
    igashell = IgAShell.IGAShell(collect(1:getncells(grid)), reverse(nurbsmesh.IEN, dims=1), igashelldata) 

    return grid, igashell
end

function get_test_mesh(CELLSTATE::IgAShell.CELLSTATE, nelx, damage)

    data = ProblemData(
        dim = 2,
        tend = 1.0,
        adaptive = true
    )

    ORDERS = (2,2)
    angles = deg2rad.([0.0, 0.0, 0.0, 0.0])
    nlayers = length(angles)
    ninterfaces = nlayers-1

    material(_α) = MatTransvLinearElastic(E1 = 61.65e3, E2 = 61.65e3, E3 = 13.61e3,ν_12 = 0.3187, ν_13 = 0.3161, ν_23 = 0.3161, G_13 = 4.55e3, G_12 = 23.37e3, G_23 = 4.55e3,α = _α) 
    layermats = [Material2D(material(α), Five.PLANE_STRAIN) for α in angles]

    nurbsmesh = IgAShell.IGA.generate_nurbs_patch(:line, (nelx, ), (ORDERS[1], ), (1.0, ), sdim=2) 
    data.grid = IgAShell.IGA.Grid(nurbsmesh)

    cellstates = [CELLSTATE for i in 1:nelx]
    
    interface_damage = damage*ones(Float64, ninterfaces, nelx)
    
    #IGAshell data
    igashelldata = 
    IgAShell.IGAShellData(;
        layer_materials           = [Material2D(material(α), Five.PLANE_STRAIN) for α in angles],
        interface_material        = Five.MatCZKolluri(σₘₐₓ = 60 * 0.5,τₘₐₓ = 90 * 0.5,Φₙ = 211.0/1000,Φₜ = 1050.0/1000),
        orders                    = ORDERS,
        knot_vectors              = nurbsmesh.knot_vectors,
        thickness                 = 1.0,
        initial_cellstates        = cellstates,
        initial_interface_damages = interface_damage,
        nqp_inplane_order         = 3,
        nqp_ooplane_per_layer     = 2,
        adaptable                 = true,
            limit_stress_criterion   = 10.993,
            limit_damage_criterion   = 0.01,
            search_radius            = 10.0,
            locked_elements          = Int[],
        small_deformations_theory = false,
        nqp_interface_order       = 4
    )  
    
    igashell = 
    IgAShell.IGAShell(
        cellset = 1:nelx, 
        data = igashelldata
    ) 
    push!(data.parts, igashell)

    data.output[] = Output(
        interval = 0.0,
        runname = "test",
        savepath = "./"
    )

    state, globaldata = build_problem(data) do dh, parts, dbc
        instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
        Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    end

    return state, globaldata

end

@testset "igashellvalues_curved" begin

    # # #
    # TEST IGASHELL VALUES
    # # #

    h = 0.2;    b = .556;     R = 4.22;
    grid, igashell = get_curved_mesh(IgAShell.LAYERED, h=h, b=b, R=R)
    addedgeset!(grid, "left", (x)-> isapprox(x[1], 0.0, atol=1e-5))
    addedgeset!(grid, "right", (x)-> x[3]≈0.0)
    addedgeset!(grid, "front", (x)-> x[2]≈0.0)
    
    rightface = collect(getedgeset(grid, "right"))
    rightedge = [IgAShell.EdgeInterfaceIndex(edgeidx..., 1) for edgeidx in rightface]
    leftface  = collect(getedgeset(grid, "left"))
    leftedge = [IgAShell.EdgeInterfaceIndex(edgeidx..., 2) for edgeidx in leftface]
    frontface  = collect(getedgeset(grid, "front"))
    frontedgetop = [IgAShell.EdgeInterfaceIndex(edgeidx..., 2) for edgeidx in frontface]
    frontedgebot = [IgAShell.EdgeInterfaceIndex(edgeidx..., 1) for edgeidx in frontface]

    #Volume
    V = 0.0
    for cellid in 1:getncells(grid)

        cv = get_and_reinit_cv(igashell, grid, cellid)

        V += calculate_element_volume(cv)
    end
    @test isapprox(V, pi/2 * R * b * h, atol=1e-3)

    #Edgelength 1
    L = 0.0
    for edge in rightedge
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge)
    end
    @test L ≈ b

    #Edgelength 2
    L = 0.0
    for edge in leftedge
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge)
    end
    @test L ≈ b

    #Edgelength 3
    L = 0.0
    for edge in frontedgetop
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge)
    end
    @test isapprox(L, (R+h/2)*pi/2, atol=1e-3)
    
    #Edgelength 4

    L = 0.0
    for edge in frontedgebot
        cellid, edgeid, interface = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        L += calculate_element_area(cv, edge)
    end
    @test isapprox(L, (R-h/2)*pi/2, atol=1e-3)

    #Side area 1
    A = 0.0
    for edge in rightface
        cellid, edgeid = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        A  += calculate_element_area(cv,edge)
    end
    @test A ≈ h*b

    #Side area 2
    A = 0.0
    for edge in leftface
        cellid, edgeid = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        A  += calculate_element_area(cv, edge)
    end
    @test A ≈ h*b

    #Side area 3
    A = 0.0
    for edge in frontface
        cellid, edgeid = edge
        cv = get_and_reinit_fv(igashell, grid, edge)

        A  += calculate_element_area(cv, edge)
    end
    @test isapprox(A, R*pi/2 * h, atol=1e-3)

    #Check curvature
    cv = get_and_reinit_cv(igashell, grid, getncells(grid)÷2)
    κ = getindex.(cv.κᵐ,1,1)
    @test all( isapprox.(κ, 1/R, atol=1e-2) )
    
end

@testset "igashell utils" begin

    order = 2;
    ninterfaces = 1
    @test (IgAShell.generate_knot_vector(order, ninterfaces, 0) .≈ Float64[-1,-1,-1, 1,1,1]) |> all
    @test (IgAShell.generate_knot_vector(order, ninterfaces, 3) .≈ Float64[-1,-1,-1, 0,0,0, 1,1,1]) |> all
    @test (IgAShell.generate_knot_vector(order, ninterfaces, 4) .≈ Float64[-1,-1,-1, 0,0,0,0, 1,1,1]) |> all

    ninterfaces = 2
    @test (IgAShell.generate_knot_vector(order, ninterfaces, 1) .≈ Float64[-1,-1,-1, -1/3, 1/3, 1,1,1]) |> all

    ninterfaces = 2
    @test (IgAShell.generate_knot_vector(order, ninterfaces, [1,2]) .≈ Float64[-1,-1,-1, -1/3, 1/3,1/3, 1,1,1]) |> all
    @test (IgAShell.generate_knot_vector(order, ninterfaces, [2,1]) .≈ Float64[-1,-1,-1, -1/3, -1/3,1/3, 1,1,1]) |> all

    ninterfaces = 3
    @test (IgAShell.generate_knot_vector(order, ninterfaces, [1,1,2]) .≈ Float64[-1,-1,-1, -0.5, 0.0, 0.5,0.5, 1,1,1]) |> all

    order = 1; ninterfaces = 3
    @test (IgAShell.generate_knot_vector(IgAShell.LUMPED_CPSTATE, order, ninterfaces) .≈ Float64[-1,-1, 1,1]) |> all
    @test (IgAShell.generate_knot_vector(IgAShell.LAYERED_CPSTATE, order, ninterfaces) .≈ Float64[-1,-1, -5/10, 0, 5/10, 1,1]) |> all
    @test (IgAShell.generate_knot_vector(IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1), order, ninterfaces) .≈ Float64[-1,-1, -5/10, -5/10, 1,1]) |> all
    @test (IgAShell.generate_knot_vector(IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(3), order, ninterfaces) .≈ Float64[-1,-1, 5/10, 5/10, 1,1]) |> all
    @test (IgAShell.generate_knot_vector(IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(2), order, ninterfaces) .≈ Float64[-1,-1, -0.5, 0.0,0.0, 0.5, 1,1]) |> all

    ##
    dim_s = 3;
    ooplane_order = 2; 
    nlayers = 2; ninterfaces = nlayers-1;
    @test 9 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.LUMPED_CPSTATE)
    @test 15 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.LAYERED_CPSTATE)
    @test 18 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1))
    @test 18 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1))

    ##
    dim_s = 3;
    ooplane_order = 1; 
    nlayers = 4; ninterfaces = nlayers-1;
    @test  6 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.LUMPED_CPSTATE)
    @test 15 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.LAYERED_CPSTATE)
    @test 18 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1))
    @test 18 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(2))
    @test 21 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,2)))
    @test 18 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(3))
    @test 12 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(3))
    @test 12 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(2))
    @test 18 == IgAShell.ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,2)))

    ## Bezier extraction matrix
    nlayers = 3; order = 1
    knot_lumped = IgAShell.generate_knot_vector(order, nlayers-1, 0)

    #lumped to layered
    new_knots = [-1 + 2i/(nlayers) for i in 1:(nlayers-1)]
    Cmat_lu2la = IgAShell.generate_out_of_plane_extraction_operators(knot_lumped, order, new_knots, fill(order, nlayers-1))

    #lumped to discont
    knot_layered = IgAShell.generate_knot_vector(order, nlayers-1, order)
    Cmat_la2di = IgAShell.generate_out_of_plane_extraction_operators(knot_layered, order, new_knots, [1,0])
    @test all(Cmat_la2di*Cmat_lu2la .≈ [1.0 0.0; 2/3 1/3; 2/3 1/3; 1/3 2/3; 0.0 1.0])

    #Multiplicity vector
    ninterfaces = 4; order = 2
    @test ([3,0,0,0] .== IgAShell.generate_nmultiplicity_vector(IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1), ninterfaces, order)) |> all
    @test ([3,2,2,2] .== IgAShell.generate_nmultiplicity_vector(IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1), ninterfaces, order)) |> all

    ## Adaptivity stuff
    h = 0.2;    b = .556;     R = 4.22;
    grid, igashell = get_curved_mesh(IgAShell.LAYERED, h=h, b=b, R=R)

    IgAShell.get_upgrade_operator(IgAShell.adapdata(igashell), IgAShell.LUMPED_CPSTATE, IgAShell.LAYERED_CPSTATE)
    IgAShell.get_upgrade_operator(IgAShell.adapdata(igashell), IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1), IgAShell.FULLY_DISCONTINIUOS_CPSTATE)
    IgAShell.get_upgrade_operator(IgAShell.adapdata(igashell), IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1), IgAShell.FULLY_DISCONTINIUOS_CPSTATE)
    IgAShell.get_upgrade_operator(IgAShell.adapdata(igashell), IgAShell.LUMPED_CPSTATE, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1))
    IgAShell.get_upgrade_operator(IgAShell.adapdata(igashell), IgAShell.LUMPED_CPSTATE, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(2))
    IgAShell.create_upgrade_operator(2, [-1/3, 1/3], IgAShell.LUMPED_CPSTATE, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(2))
    #IgAShell.@showm IgAShell.IgAShell.IGA.beo2matrix(C)'

    #Active basefunctions in layer
    order = 2; ilay = 1; ninterfaces=2
    @test (IgAShell.get_active_basefunctions_in_layer(1, order, IgAShell.LUMPED_CPSTATE) .== 1:order+1) |> all
    @test (IgAShell.get_active_basefunctions_in_layer(2, order, IgAShell.LUMPED_CPSTATE) .== 1:order+1) |> all

    @test (IgAShell.get_active_basefunctions_in_layer(1, order, IgAShell.LAYERED_CPSTATE) .== 1:order+1)  |> all
    @test (IgAShell.get_active_basefunctions_in_layer(2, order, IgAShell.LAYERED_CPSTATE) .== (1:order+1) .+ order) |> all

    @test (IgAShell.get_active_basefunctions_in_layer(1, order, IgAShell.FULLY_DISCONTINIUOS_CPSTATE) .== (1:order+1)) |> all
    @test (IgAShell.get_active_basefunctions_in_layer(2, order, IgAShell.FULLY_DISCONTINIUOS_CPSTATE) .== (1:order+1) .+ (order+1)) |> all

    @test (IgAShell.get_active_basefunctions_in_layer(2, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,))) .== (1:order+1) .+ (order)) |> all
    @test (IgAShell.get_active_basefunctions_in_layer(3, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,))) .== (1:order+1) .+ 5) |> all

    @test (IgAShell.get_active_basefunctions_in_layer(1, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,))) .== (1:order+1)) |> all
    @test (IgAShell.get_active_basefunctions_in_layer(2, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,))) .== (1:order+1) .+ order.+1) |> all
    @test (IgAShell.get_active_basefunctions_in_layer(3, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,))) .== (1:order+1) .+ order.+1) |> all

    #
    order = 2; ninterfaces=3
    IgAShell.insert_interface(IgAShell.LUMPED_CPSTATE, 1, ninterfaces) == IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(1)
    IgAShell.insert_interface(IgAShell.LAYERED_CPSTATE, 2, ninterfaces) == IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(2)
    IgAShell.insert_interface(IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,3)), 1, ninterfaces) == IgAShell.FULLY_DISCONTINIUOS_CPSTATE
    IgAShell.insert_interface(IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,3)), 1, ninterfaces) == IgAShell.FULLY_DISCONTINIUOS_CPSTATE

    #
    IgAShell.insert_interface(IgAShell.LUMPED, 2, ninterfaces).state == IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE((2,)).state
    IgAShell.insert_interface(IgAShell.LAYERED, 2, ninterfaces).state == IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE((2,)).state

    ninterfaces = 1
    IgAShell.insert_interface(IgAShell.LAYERED, 1, ninterfaces).state == IgAShell.FULLY_DISCONTINIUOS.state


    #
    order = 2
    (IgAShell.get_active_basefunctions_in_interface(2, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,))) .== [3,4]) |> all
    (IgAShell.get_active_basefunctions_in_interface(1, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,))) .== [1,2,3]) |> all
    (IgAShell.get_active_basefunctions_in_interface(3, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,3,))) .== [6,7]) |> all

    (IgAShell.get_active_basefunctions_in_interface(1, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,3,))) .== [3,4]) |> all
    (IgAShell.get_active_basefunctions_in_interface(2, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,3,))) .== [4,5,6]) |> all
    (IgAShell.get_active_basefunctions_in_interface(3, order, IgAShell.WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,3,))) .== [6,7]) |> all

    (IgAShell.get_active_basefunctions_in_interface(1, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,3,))) .== [3,4]) |> all
    (IgAShell.get_active_basefunctions_in_interface(2, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,3,))) .== [6]) |> all
    (IgAShell.get_active_basefunctions_in_interface(3, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((1,3,))) .== [8,9]) |> all

    (IgAShell.get_active_basefunctions_in_interface(1, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,3,))) .== [3]) |> all
    (IgAShell.get_active_basefunctions_in_interface(2, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,3,))) .== [5,6]) |> all
    (IgAShell.get_active_basefunctions_in_interface(3, order, IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE((2,3,))) .== [8,9]) |> all
end

@testset "stiffness_matrix_lumped" begin
    state, globaldata = get_test_mesh(IgAShell.LUMPED, 1, 0.0)

    fill!(state.system_arrays, 0.0)
    Five.assemble_stiffnessmatrix_and_forcevector!(globaldata.dh, state, globaldata)
    @test norm(state.system_arrays.Kⁱ) ≈ 79420.08082130156
end

@testset "stiffness_matrix_layered" begin
    state, globaldata = get_test_mesh(IgAShell.LAYERED, 1, 0.0)

    fill!(state.system_arrays, 0.0)
    Five.assemble_stiffnessmatrix_and_forcevector!(globaldata.dh, state, globaldata)
    @test norm(state.system_arrays.Kⁱ) ≈ 159622.44106964912
end

@testset "stiffness_matrix_discont" begin
    state, globaldata = get_test_mesh(IgAShell.FULLY_DISCONTINIUOS, 1, 0.0)

    fill!(state.system_arrays, 0.0)
    Five.assemble_stiffnessmatrix_and_forcevector!(globaldata.dh, state, globaldata)
    @test norm(state.system_arrays.Kⁱ) ≈ 158770.59191158036
end

@testset "stiffness_matrix_discont_with_damage" begin
    state, globaldata = get_test_mesh(IgAShell.FULLY_DISCONTINIUOS, 1, 0.5)

    fill!(state.system_arrays, 0.0)
    Five.assemble_stiffnessmatrix_and_forcevector!(globaldata.dh, state, globaldata)
    @test norm(state.system_arrays.Kⁱ) ≈ 138016.89592686333
end

@testset "stiffness_matrix_discont_with_damage_nelx3" begin
    state, globaldata = get_test_mesh(IgAShell.FULLY_DISCONTINIUOS, 3, 0.5)

    fill!(state.system_arrays, 0.0)
    Five.assemble_stiffnessmatrix_and_forcevector!(globaldata.dh, state, globaldata)
    @test norm(state.system_arrays.Kⁱ) ≈ 184212.0377700771
end

#=
@testset "igashell values" begin 

#    grid, igashell = get_cube_mesh(IgAShell.LUMPED, h=0.1, b=1.0, L = 10.0)
    h = 0.2
    grid, igashell = get_curved_mesh(IgAShell.LAYERED, h=h, b=.556, R=4.22)
    dim = Ferrite.getdim(igashell)

    for cellid in 1:1#getncells(igashell)
        cellstate = IgAShell.getcellstate(IgAShell.adapdata(igashell), cellid)

        cv = IgAShell.build_cellvalue!(igashell, cellid)
        Ce = IgAShell.get_extraction_operator(IgAShell.intdata(igashell), cellid)
        active_layerdofs = IgAShell.build_active_layer_dofs(igashell, cellstate)
        ndofs_per_cell = maximum(maximum.(active_layerdofs))

        IgAShell.IgAShell.IGA.set_bezier_operator!(cv, Ce)

        coords = getcoordinates(grid, cellid)
        bezier_coords = IgAShell.IGA.compute_bezier_points(Ce, coords)
        
        IgAShell.reinit_midsurface!(cv, bezier_coords)
        
        ue = rand(Float64, ndofs_per_cell)*0.001
        
        for ilay in 1:IgAShell.nlayers(igashell)
            IgAShell.reinit_layer!(cv, ilay)
            ue_layer = ue[active_layerdofs[ilay]]
            for qp in 1:IgAShell.getnquadpoints_per_layer(cv)
                #dU
                dUdξ = gradient( (ξ) -> IgAShell._calculate_u((cv.inp_ip, cv.oop_ip, Ce), ue, ξ), cv.qr[ilay].points[qp])
                for d in 1:dim
                    dU1 = IgAShell.function_parent_derivative(cv, qp, ue_layer, d)
                    dU2 = dUdξ[:,d]
                    @test dU1 ≈ dU2
                end
                #G
                for d in 1:dim
                    G = IgAShell._calculate_G(cv.inp_ip, bezier_coords, h, cv.qr[ilay].points[qp], d)
                    @test cv.G[qp][d] ≈  G
                end
                #g
                for d in 1:dim
                    g1 = IgAShell._calculate_g((cv.inp_ip, cv.oop_ip, Ce), bezier_coords, h, ue, cv.qr[ilay].points[qp], d)
                    g2 = cv.G[qp][d] + IgAShell.function_parent_derivative(cv, qp, ue_layer, d)
                    @test g1 ≈ g2
                end

                for i in 1:IgAShell.getnbasefunctions_per_layer(cv)
                    ξ = cv.qr[ilay].points[qp]

                    N(_ξ) = IgAShell._calculate_basefunc((cv.inp_ip, cv.oop_ip, Ce), active_layerdofs[ilay][i], _ξ)
                    @test N(ξ) ≈ cv.N[i, qp]

                    dNdξ = gradient( (_ξ) -> N(_ξ), cv.qr[ilay].points[qp])
                    @test dNdξ ≈ cv.dNdξ[i, qp]
                end

            end
        end
    end


end
=#