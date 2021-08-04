
function _commit_part!(dh::Ferrite.AbstractDofHandler, 
                      igashell::IGAShell{dim_p,dim_s}, 
                      state::StateVariables) where {dim_p,dim_s}

    instructions = FieldDimUpgradeInstruction[]
    upgraded_cells = Int[]
    cellnodes = zeros(Int, Ferrite.nnodes_per_cell(igashell))
    #copy of the current node states
    node_states = deepcopy(adapdata(igashell).control_point_states)

    #
    # Loop through all elements and check damage variables for stress propagation
    #
    for (ic, cellid) in enumerate(igashell.cellset)

        cellid in igashell.layerdata.locked_elements && continue

        cellstate = getcellstate(adapdata(igashell), ic)
        
        cell_material_states = state.partstates[ic]

        for iint in 1:ninterfaces(igashell)
            
            if igashell.adaptivity.propagation_checked[iint, ic] == true
                continue
            end

            HARDCODED_DIR = 1
            interface_damage_variables = interface_damage.(cell_material_states.interfacestates[:, iint], HARDCODED_DIR)

            upgrade_list = determine_crack_growth(dh, igashell, interface_damage_variables, cellid, iint)

            if length(upgrade_list) > 0
                println("Propagation was determined for $cellid, interface $iint, $(mean(interface_damage_variables))")
            end

            for nodeid in upgrade_list

                nodeid in igashell.adaptivity.locked_control_points && continue

                igashell.adaptivity.propagation_checked[iint, ic] = true

                node_states[nodeid] = insert_interface(node_states[nodeid], iint, ninterfaces(igashell))
            end
        end
    end

    #
    # Loop through all elements, and check which elements need to upgrade based on the stress values
    #
    for (ic, cellid) in enumerate(igashell.cellset)

        cellid in igashell.layerdata.locked_elements && continue

        cellstate = getcellstate(adapdata(igashell), ic)
        recovory_stress_data = @view srdata(igashell).recovered_stresses[:,ic]
        constitutive_stress_data = @view igashell.integration_data.interfacestresses[:,ic]#state.partstates[ic].materialstates

        _should_upgrade, upgrade_to, interface_to_upgrade = determine_upgrade(igashell, recovory_stress_data, constitutive_stress_data, cellstate)

        if _should_upgrade
            println("Initiation was determined for cell $cellid, interfaces: $(interface_to_upgrade)")

            # Loop through all nodes for this cell and set the state 
            # of the node to the state of the cell, if it is an "upgrade"
            Ferrite.cellnodes!(cellnodes, dh, cellid)
            for (i, nodeid) in enumerate(cellnodes)
                node_states[nodeid] = combine_states(node_states[nodeid], get_cpstate(upgrade_to, i), ninterfaces(igashell))
            end
        end
    end

    # Second loop over all the cells
    # Some of the elements will now be in a mixed state.
    # Find these, and set the state of these cells to MIXED
    # Also, create the "instruction" for the dofhandler on how to upgrade dofs for the cells
    for (ic, cellid) in enumerate(igashell.cellset)
        _celldofs = Ferrite.celldofs(dh, cellid)
        ue = state.d[_celldofs]
        Δue = state.Δd[_celldofs]
        
        Ferrite.cellnodes!(cellnodes, dh, cellid)
        new_cellnode_states = node_states[cellnodes]
        current_cellnode_states = igashell.adaptivity.control_point_states[cellnodes]

        # Check if any of the nodes in the cell has been upgraded this timestep
        # If not, no upgraded is needed.
        if all(new_cellnode_states .== current_cellnode_states)
            continue
        end

        instr = construct_upgrade_instruction(igashell, cellid, cellnodes, current_cellnode_states, new_cellnode_states, ue, Δue)
        push!(instructions, instr)

        # Check if the element is mixed.
        if all(new_cellnode_states[1] .== new_cellnode_states)
            setcellstate!(adapdata(igashell), cellid, CELLSTATE(new_cellnode_states[1].state, new_cellnode_states))
        else
            #Get integer represenation of active interfaces
            setcellstate!(adapdata(igashell), cellid, CELLSTATE(_MIXED, new_cellnode_states))
        end

    end

    adapdata(igashell).control_point_states .= node_states

    return instructions
end

function determine_crack_growth(dh::MixedDofHandler, igashell::IGAShell, interface_damage_variables::Vector{Float64}, cellid::Int, iint::Int)

    cp_to_upgrade = Int[]

    ncontroloints = length(igashell.adaptivity.control_point_states)
    nnodes = Ferrite.nnodes_per_cell(igashell)
    cellnodes = zeros(Int, nnodes)
    
    #
    #Check propation
    #
    mean_damage = mean(interface_damage_variables)
    if mean_damage > LIMIT_DAMAGE_VARIABLE(layerdata(igashell))

        #Get all nodeids in this cell,
        Ferrite.cellnodes!(cellnodes, dh, cellid)
        
        #Loop over all nodes in the cell
        for cpid in cellnodes

            #Get position of the node
            pos1 = dh.grid.nodes[cpid].x

            #Loop over all controlpoints to see if they are withing search radius
            for jcp in 1:ncontroloints
                cp_state = get_controlpoint_state(adapdata(igashell), jcp)

                #Skip if alrady active interface
                if is_interface_active(cp_state, iint)
                    continue
                end

                
                #Check if controlpoint is within search radius
                if norm(pos1 - dh.grid.nodes[jcp].x) < PROPAGATION_SEARCH_RADIUS(layerdata(igashell))
                    #Skip if this node is in the cell we are currently uppgrading
                    if jcp in cellnodes
                        continue
                    end
                    push!(cp_to_upgrade, jcp)
                end
            end
        end
    end

    return cp_to_upgrade

end

function determine_upgrade(igashell::IGAShell{dim_p, dim_s, T}, 
                           recovory_stress_data::AbstractVector{<:RecoveredStresses}, 
                           constitutive_stress_data::AbstractVector{SymmetricTensor{2,3,Float64,6}}, 
                           cellstate::CELLSTATE) where {dim_p, dim_s, T}

    is_fully_discontiniuos(cellstate) && return false, nothing, Int[]
    imat = interface_material(igashell)
    cell_upgraded = false
    new_cellstate = cellstate
    interface_to_upgrade = Int[]

    τᴹ = Five.max_traction_force(imat, 1)
    σᴹ = Five.max_traction_force(imat, dim_s)

    #
    nqp_oop_per_layer = getnquadpoints_ooplane_per_layer(igashell)
    nqp_oop = getnquadpoints_ooplane(igashell)
    nqp_inp = getnquadpoints_inplane(igashell)

    for iint in 1:ninterfaces(igashell)
        
        #If interface already delaminated, ignore this interface
        if is_interface_active(cellstate, iint) && !is_mixed(cellstate) 
            continue
        end

        #Calculate stress at interface, midle od elements
        local σᶻˣ, σᶻʸ, σᶻᶻ 
        if is_lumped(cellstate)
            σᶻˣ, σᶻʸ, σᶻᶻ = _get_interface_stress_lumped(recovory_stress_data, iint, nqp_oop_per_layer)
        else
            σᶻˣ, σᶻʸ, σᶻᶻ = _get_interface_stress_layered(constitutive_stress_data, iint)
        end
        

        macl(x) = (x<=0 ? (0.0) : x)
        F = (σᶻˣ/ τᴹ)^2 + (σᶻʸ / τᴹ)^2 + (macl(σᶻᶻ) / σᴹ)^2

        if F > LIMIT_STRESS_CRITERION(layerdata(igashell))
            #@show σᶻˣ, σᶻʸ, σᶻᶻ
            new_cellstate = insert_interface(new_cellstate, iint, ninterfaces(igashell))
            append!(interface_to_upgrade, iint)
            cell_upgraded = true
        end
    end

    return cell_upgraded, new_cellstate, interface_to_upgrade
end

function _get_interface_stress_layered(materialstates::AbstractVector{SymmetricTensor{2,3,Float64,6}}, iint)

    σ = materialstates[iint]
    #Use dim_p to make to code work in both 2d and 3d
    return σ[1,3], σ[2,3], σ[3,3]
end

function _get_interface_stress_lumped(stress_data::AbstractVector{<:RecoveredStresses}, iint::Int, nqp::Int)

    #Get the index which corresponds to the stress at the interface, in the recovered_stress-vector
    idx = recovered_stress_interface_index(iint, nqp)

    #
    σᶻˣ = stress_data[idx].σᶻˣ
    σᶻʸ = stress_data[idx].σᶻʸ
    σᶻᶻ = stress_data[idx].σᶻᶻ
    return σᶻˣ, σᶻʸ, σᶻᶻ
end


"""
stress_interface_index
    Calculates the index for the mid quadraturepoint in the materialstate matrix, given a specific qplayerid
    If even number of quadraturepoints inplane, round to nearest quadpoint in the middle. 
    The materialstate-matrix orders the quadraturepoints states by  [ layer, [qpinplane(xdir) x qpinplane(ydir) x qpooplane(zdir)] ] 
"""
function stress_interface_index(qplayerid::Int, nqpinplane::Int, dim_p::Int)
    qp_order = round(Int, nqpinplane^(1/dim_p))
    row = ceil(Int, qp_order/2)
    idx = nqpinplane*(qplayerid-1) + qp_order*(row-1)*(dim_p-1) + row
    return idx
end

function construct_upgrade_instruction(igashell::IGAShell{dim_p,dim_s,T}, cellid::Int, cellnodes::Vector{Int}, current_cellnode_state::AbstractVector{CPSTATE}, cellnode_states::AbstractVector{CPSTATE}, ue::Vector{T}, Δue::Vector{T}) where {dim_p,dim_s,T}

    local_dof_idxs = Int[]
    current_fielddims = Int[]
    update_fielddims = Int[]
    extended_ue = Vector{Float64}[]
    extended_Δue = Vector{Float64}[]
    nodeids = Int[]

    local_dof_idx = 1

    # Loop through all node in the cell and create the "instructions" that the 
    # dofhandler needs in order to distribute new dofs.

    for (i, nodeid) in enumerate(cellnodes)

        cp_state = current_cellnode_state[i]
        current_fielddim = ndofs_per_controlpoint(igashell, cp_state)

        # Check if the controlpoint is being "upgraded"
        if cellnode_states[i] == cp_state || cellnode_states[i].state < cp_state.state
            local_dof_idx += current_fielddim
            continue
        end
        
        updated_cellstate = cellnode_states[i]
        update_fielddim = ndofs_per_controlpoint(igashell, updated_cellstate)

        # Calculate the values for the new controlpoints
        C = get_upgrade_operator(adapdata(igashell), cp_state, updated_cellstate)

        ue_node = ue[(1:current_fielddim) .+ (local_dof_idx-1)]
        ue_extended = IGA.compute_bezier_points(C, ue_node, dim=dim_s)
        
        Δue_node = Δue[(1:current_fielddim) .+ (local_dof_idx-1)]
        Δue_extended = IGA.compute_bezier_points(C, Δue_node, dim=dim_s)

        local_dof_idx += current_fielddim

        push!(local_dof_idxs, local_dof_idx)
        push!(current_fielddims, current_fielddim)
        push!(update_fielddims, update_fielddim)
        push!(extended_ue, ue_extended)
        push!(extended_Δue, Δue_extended)
        push!(nodeids, nodeid)
    end

    instr = FieldDimUpgradeInstruction(cellid, nodeids, local_dof_idxs, #What cell, and what dof idx
                                        current_fielddims, update_fielddims, # number of dofs
                                        extended_ue, extended_Δue) #the new values pushed into the state vector

    return instr

end