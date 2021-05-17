

function IGAShellVTK(data::IGAShellData{dim_p,dim_s,T}) where {dim_p,dim_s,T}

    n_plot_points_dim = dim_s==2 ? (2,2) : (2,2,2)
    @assert prod(n_plot_points_dim .- 1) == 1

    mid_ip = getmidsurface_ip(data)
    _refcoords = Ferrite.reference_coordinates(IGA.BernsteinBasis{dim_p,n_plot_points_dim[1:dim_p].-1}())
    plot_point_inplane = QuadratureRule{dim_p,RefCube,T}(zeros(T,length(_refcoords)), _refcoords)
    
    order = data.orders[dim_s]
    knot_discont = generate_knot_vector(order, data.nlayers-1, order+1)
    bbasis_discont = IGA.BSplineBasis(knot_discont, order)

    #The zcoord to plot for each layer
    zcoords = data.zcoords
    addon = (last(zcoords) + first(zcoords))/2
    scale = (last(zcoords) - first(zcoords))/2
    zcoords = (zcoords.-addon)/scale
    zcoords = collect(Iterators.flatten(zip(zcoords.- 1e-13, zcoords)))[2:end-1] 
    plot_points_ooplane = QuadratureRule{1,RefCube,T}(zeros(T,length(zcoords)), [Vec((z,)) for z in zcoords])
    discont_values = BasisValues(plot_points_ooplane, bbasis_discont)

    cv = IGAShellValues(data.thickness, plot_point_inplane, plot_points_ooplane, mid_ip, getnbasefunctions(bbasis_discont))
    set_oop_basefunctions!(cv, discont_values)

    return IGAShellVTK{dim_p,dim_s,T,typeof(cv)}(cv, n_plot_points_dim, Vec{dim_s,T}[], MeshCell{VTKCellType, Vector{Int}}[])
end

nvtkcells_per_layer(vtkdata::IGAShellVTK) = prod(vtkdata.n_plot_points_dim .- 1)::Int
nvtkcells_per_interface(vtkdata::IGAShellVTK{dim_p}) where {dim_p} = prod(vtkdata.n_plot_points_dim[1:dim_p] .- 1)::Int

"""
    _init_vtk_grid!

Calculates the coordinates and cells for the underlying FE-mesh that is used to visualize the 
shell in VTK.

The code is not very pretty, and is mostly a hack...

"""
function _init_vtk_grid!(dh::Ferrite.AbstractDofHandler, igashell::IGAShell{dim_p,dim_s,T}) where {dim_p,dim_s,T}

    node_coords = Vec{dim_s,T}[]
    cls = MeshCell[]
    node_offset = 0
    coords = zeros(Vec{dim_s,T}, Ferrite.nnodes_per_cell(igashell))
    bezier_coords = similar(coords)
    cv_plot = vtkdata(igashell).cell_values_plot
    n_plot_points_dims = vtkdata(igashell).n_plot_points_dim

    cellcount = 0
    for (ic, cellid) in enumerate(igashell.cellset)
        
        Ce = get_extraction_operator(intdata(igashell), ic)
        
        Ferrite.cellcoords!(coords, dh, cellid)
        bezier_coords .= IGA.compute_bezier_points(Ce, coords)
        
        IGA.set_bezier_operator!(cv_plot, Ce)
        reinit!(cv_plot, bezier_coords)

        _init_vtk_cell!(igashell, cls, node_coords, vtkdata(igashell).cell_values_plot, bezier_coords, node_offset, n_plot_points_dims)
        cellcount = length(cls)
        node_offset = length(node_coords)
    end

    Ferrite.copy!!(igashell.vtkdata.node_coords, copy(node_coords))
    Ferrite.copy!!(igashell.vtkdata.cls, copy(cls))
end

function _init_vtk_cell!(igashell::IGAShell{dim_p,dim_s}, cls, node_coords, cv, bezier_coords::Vector{Vec{dim_s,T}}, node_offset::Int, n_plot_points_dims) where {dim_p,dim_s,T}

    for point_id in 1:getnquadpoints(cv)
        X = spatial_coordinate(cv, point_id, bezier_coords)
        push!(node_coords, X)
    end

    n_plot_points = prod(n_plot_points_dims)::Int
    n_plot_points_inp = prod(n_plot_points_dims[1:dim_p])::Int
    n_plot_points_oop = n_plot_points_dims[dim_s]
    nodeind2nodeid = reshape(1:n_plot_points, n_plot_points_dims)
    indeces = CartesianIndices(n_plot_points_dims .- 1 )[:]
    addons  = CartesianIndices(Tuple(fill(2,dim_s)))[:]

    #Build the cells based on the plot points
    
    for ilay in 1:nlayers(igashell)
        for index in indeces
            
            nodeids = Int[]
            for a in 1:2^dim_s
                nodendex = Tuple(index).+ Tuple(addons[a]) .-1
                _nodeid = nodeind2nodeid[nodendex...]
                push!(nodeids, _nodeid)
            end

            #Shell layer
            VTK_CELL = (dim_s==2) ? Ferrite.VTKCellTypes.VTK_QUAD : Ferrite.VTKCellTypes.VTK_HEXAHEDRON
            nodeids = (dim_s==2) ? nodeids[[1,2,4,3]] : nodeids[[1,2,4,3,5,6,8,7]]
            push!(cls, MeshCell(VTK_CELL, nodeids .+ node_offset .+ (ilay-1)*n_plot_points))

        end 
    end

    if igashell.layerdata.add_czcells_vtk
        for iint in 1:ninterfaces(igashell)
            for index in indeces
                nodeids = Int[]
                for a in 1:2^dim_s
                    nodendex = Tuple(index).+ Tuple(addons[a]) .-1
                    _nodeid = nodeind2nodeid[nodendex...]
                    push!(nodeids, _nodeid)
                end

                #Cohesive
                VTK_CELL = (dim_s==2) ? Ferrite.VTKCellTypes.VTK_QUAD : Ferrite.VTKCellTypes.VTK_HEXAHEDRON
                localids = (dim_s==2) ? [3,4] : [5,6,8,7]
                nodeids = vcat(nodeids[localids], nodeids[localids] .+ n_plot_points_inp*(n_plot_points_oop-1))
                
                push!(cls, MeshCell(VTK_CELL, nodeids .+ node_offset .+ (iint-1)*n_plot_points))
            end
        end
    end
    
    
end

"""
    get_vtk_displacements

Calculates the dispalcements at the nodes of the underlying FE-mesh 

"""
function Five.get_vtk_displacements(dh::Ferrite.AbstractDofHandler, igashell::IGAShell{dim_p,dim_s,T}, state::StateVariables) where {dim_p,dim_s,T}


    node_coords = Vec{dim_s,T}[]
    cls = MeshCell[]
    node_offset = 0
    nodes = zeros(Int, Ferrite.nnodes_per_cell(igashell))
    X = zeros(Vec{dim_s,T}, Ferrite.nnodes_per_cell(igashell))
    Xᵇ = similar(X)
    cv_plot = vtkdata(igashell).cell_values_plot
    node_disps = Vec{dim_s,T}[]
    celldofs = zeros(Int, Ferrite.ndofs_per_cell(dh,1))
    for (ic, cellid) in enumerate(igashell.cellset)
        
        Ce = get_extraction_operator(intdata(igashell), ic)
        
        Ferrite.cellcoords!(X, dh, cellid)
        Xᵇ .= IGA.compute_bezier_points(Ce, X)

        IGA.set_bezier_operator!(cv_plot, Ce)
        reinit!(cv_plot, Xᵇ)

        #Get displacement for this cell
        ndofs = Ferrite.ndofs_per_cell(dh,cellid)
        resize!(celldofs, ndofs)
        Ferrite.celldofs!(celldofs, dh, cellid)
        ue = state.d[celldofs]

        #Transform displayements to fully discontinuous state
        Ferrite.cellnodes!(nodes, dh, ic)
        ue_bezier = T[]
        offset = 1
     
        @timeit "nodes" for nodeid in nodes
            current_state = get_controlpoint_state(adapdata(igashell), nodeid)
            
            _ndofs = ndofs_per_controlpoint(igashell, current_state)

            if is_fully_discontiniuos(current_state)
                append!(ue_bezier, ue[offset:offset+_ndofs-1])
                offset += _ndofs
                continue
            end

            _Coop = get_upgrade_operator(adapdata(igashell), current_state, FULLY_DISCONTINIUOS_CPSTATE)
            
            _ue_bezier = IGA.compute_bezier_points(_Coop, ue[offset:offset+_ndofs-1], dim = dim_s)

            append!(ue_bezier, _ue_bezier)
            offset += _ndofs
        end

        @timeit "funcvalue" for point_id in 1:getnquadpoints(cv_plot)
            u = function_value(cv_plot, point_id, ue_bezier)
            push!(node_disps, u)
        end

    end
    
    return node_disps

end

"""
    get_vtk_celldata

Gets the cellstate in each underlying FE-mesh cell
"""
function Five.get_vtk_celldata(igashell::IGAShell{dim_p,dim_s,T}, output::VTKCellOutput{<:IGAShellConfigStateOutput}, state::StateVariables, globaldata) where {dim_p,dim_s,T}

    vtkcellcount = 1
    nvtkcells = nvtkcells_per_layer(vtkdata(igashell)) * nlayers(igashell) * getncells(igashell) #+ 
    
    if igashell.layerdata.add_czcells_vtk 
        nvtkcells +=  nvtkcells_per_interface(vtkdata(igashell)) * ninterfaces(igashell) * getncells(igashell)
    end

    cellstates = zeros(Int, nvtkcells)
    cellnodes = zeros(Int, Ferrite.nnodes_per_cell(igashell))

    for (ic, cellid) in enumerate(igashell.cellset)
        cellstate = getcellstate(adapdata(igashell), ic)
        for ilay in 1:nlayers(igashell)
            for vtkcell in nvtkcells_per_layer(vtkdata(igashell))
                
                Ferrite.cellnodes!(cellnodes, globaldata.dh, ic)
                tmp_cellstate = get_controlpoint_state(adapdata(igashell), cellnodes[1])
                for (i, nodeid) in enumerate(cellnodes[2:end])
                    cp_state = get_controlpoint_state(adapdata(igashell), nodeid)
                    tmp_cellstate = tmp_cellstate.state > cp_state.state ? tmp_cellstate : cp_state
                end
                cellstates[vtkcellcount] = get_cellstate_color(tmp_cellstate)
                vtkcellcount += 1
            end
        end
        if igashell.layerdata.add_czcells_vtk
            for iint in 1:ninterfaces(igashell)
                state_color = -1
                if is_interface_active(cellstate, iint)
                    state_color = 6
                end

                for vtkcell in nvtkcells_per_layer(vtkdata(igashell))
                    cellstates[vtkcellcount] = state_color
                    vtkcellcount += 1
                end
            end
        end
    end

    return cellstates

end

"""
    get_vtk_nodedata

Gets the interface damage at the nodes by taking the mean of the damage at all quadpoints.
"""
function Five.get_vtk_nodedata(igashell::IGAShell{dim_p,dim_s,T}, output::VTKNodeOutput{<:IGAShellMaterialStateOutput}, state::StateVariables, globaldata) where {dim_p,dim_s,T}

    n_damage_paras = Five.n_damage_parameters(igashell.layerdata.interface_material)
    n_vtknodes_per_layer = prod(vtkdata(igashell).n_plot_points_dim)::Int
    n_vtknodes_per_cell = n_vtknodes_per_layer * nlayers(igashell)
    n_vtknodes_inplane = prod(vtkdata(igashell).n_plot_points_dim[1:dim_p])::Int
    n_vtknodes_oop_per_layer = vtkdata(igashell).n_plot_points_dim[dim_s]
    @assert( n_vtknodes_oop_per_layer > 1)

    vtkcellcount = 1
    nvtkcells = nvtkcells_per_layer(vtkdata(igashell)) * nlayers(igashell) * getncells(igashell)
    cellstates = zeros(Int, nvtkcells)
    vtk_node_damage = fill(0.0, n_damage_paras, n_vtknodes_per_cell * getncells(igashell))
    for (ic, cellid) in enumerate(igashell.cellset)

        #Since all cells in IGAShell has its set of non connected nodes, 
        # one has to calculate different interface damage values for each cell even though the node is actually shared between cells...
        cell_offset = (ic-1) * n_vtknodes_per_cell

        for ilay in 1:ninterfaces(igashell)
            interfacestates = state.partstates[ic].interfacestates[:,ilay]

            mean_damages = zeros(T, n_damage_paras)
            for i in 1:n_damage_paras
                mean_damages[i] = mean(interface_damage.(interfacestates, i))
            end
            
            layer_offset =  (ilay-1)*n_vtknodes_per_layer  +  n_vtknodes_inplane*(n_vtknodes_oop_per_layer-1)

            for i in 1:n_vtknodes_inplane*2
                nodeidx = i + cell_offset + layer_offset
                for i in 1:n_damage_paras
                    vtk_node_damage[i, nodeidx] = mean_damages[i]
                end
            end
        end
    end
    
    return vtk_node_damage

end
