import numpy as np
import pyttb as ttb
import math
 

class ExoInfo:
    """
    Read and write Exodus-II (netCDF) “sheets,” extract a single-z planar slice of nodal data,
    build finite-element quadrature tables, and perform spatial integrals (and their derivatives)
    on those slices.

    Usage
    -----
    1. Instantiate:  exo = ExoInfo()
    2. Read a slice: exo.read_sheet('foo.e', z_slice=0.0)
         → populates:
             x_coord    : (num_selected_nodes,)   float64
             y_coord    : (num_selected_nodes,)   float64
             z_coord    : (num_selected_nodes,)   float64  (all equal to z_slice)
             x, y, z    : 1D arrays of the unique slice coordinates
             t          : (num_time,)             float64
             tensor_data: (nx, ny, num_vars, num_time) float64
             var_name   : list[str]  of length num_vars
             node_ind                : (num_selected_nodes, 3)    int (ix,iy,iz)
             tensor_node_ind         : (num_selected_nodes, 2)    int (ix,iy)
             elem_ind                : (num_elem, num_nodes_per_elem) int zero-based
             A                        : (num_qp_per_elem, num_nodes_per_elem) float64
             gp                       : (num_qp_per_elem, 2) float64
             w_det_J                  : (num_qp_per_elem, num_elem) float64
             linear_elem_ind          : (num_qp_per_elem * num_elem,) int
             node_linear_ind          : (nx*ny,) int
             tensor_elem_linear_ind   : (num_elem, num_nodes_per_elem) int
    3. Call compute_spatial_integral(...) as many times as you like.
    4. Optionally write back out via write_sheet(...).

    Attributes
    ----------
    x_coord : ndarray, shape (num_selected_nodes,)
        Full list of x-coordinates for the nodes that lie exactly at z_slice.
    y_coord : ndarray, shape (num_selected_nodes,)
    z_coord : ndarray, shape (num_selected_nodes,)
        Should all be equal to the requested z_slice.
    x : ndarray, shape (nx,)
        The unique sorted x-coordinates.
    y : ndarray, shape (ny,)
    z : ndarray, shape (1,)
        Singleton array containing exactly [z_slice].
    t : ndarray, shape (num_time,)
        The time values from the “time_whole” variable.
    tensor_data : ndarray, shape (nx, ny, num_vars, num_time)
        The dense nodal data arranged as a full 4-D array.
    var_name : list of str, length num_vars
        The names of each nodal variable.
    node_ind : ndarray, shape (num_selected_nodes,3)
        The (ix,iy,iz) indices in the full (nx,ny,nz) grid for each selected node.
    tensor_node_ind : ndarray, shape (num_selected_nodes,2)
        The (ix,iy) indices into tensor_data for each selected node.
    elem_ind : ndarray, shape (num_elem, num_nodes_per_elem)
        Zero-based element connectivity arrays.
    A : ndarray, shape (num_qp_per_elem, num_nodes_per_elem)
        Shape-function values at each quadrature point.
    gp : ndarray, shape (num_qp_per_elem,2)
        Quadrature point coordinates in the reference square [-1,+1]^2.
    w_det_J : ndarray, shape (num_qp_per_elem, num_elem)
        Quadrature weights multiplied by |det(J)| for each element.
    linear_elem_ind : ndarray, shape (num_qp_per_elem * num_elem,)
        Flattened list of global node-indices (into node_linear_ind) for every qp+elem.
    node_linear_ind : ndarray, shape (nx*ny,)
        Maps 2-D tensor indices into a flat index for assembly.
    tensor_elem_linear_ind : ndarray, shape (num_elem, num_nodes_per_elem)
        For each element, the flat 1-D index of each corner node.
    """

        
    def read_sheet(self, file_name, z_slice=0.0):
        """
        Load an Exodus-II (netCDF4) file and extract nodal data on the plane z == z_slice.

        Parameters
        ----------
        file_name : str
            Path to an Exodus-II netCDF file.
        z_slice : float, optional
            Which z-coordinate to extract.  Only nodes exactly at this z will
            be included in the planar slice.

        Raises
        ------
        AssertionError
            If no nodes are found at z == z_slice or if more than one distinct
            z-value remains after slicing.
        """
        import netCDF4 as nc
        
        # Read file
        d = nc.Dataset(file_name,'r')

        # Get dimensions
        num_var = len(d.dimensions['num_nod_var'])
        num_time = len(d.dimensions['time_step'])

        # get coordinates of each node
        self.x_coord = np.array(d.variables['coordx'])
        self.y_coord = np.array(d.variables['coordy'])
        self.z_coord = np.array(d.variables['coordz'])

        # restrict coordinates to given slice
        valid_ind = np.nonzero(self.z_coord == z_slice)[0]
        self.x_coord = self.x_coord[valid_ind]
        self.y_coord = self.y_coord[valid_ind]
        self.z_coord = self.z_coord[valid_ind]
        num_node = len(valid_ind)

        # get time steps
        self.t = np.array(d.variables['time_whole'])

        # unique coordinates forms our modes
        self.x = np.unique(self.x_coord)
        self.y = np.unique(self.y_coord)
        self.z = np.unique(self.z_coord)
        num_x = len(self.x)
        num_y = len(self.y)
        num_z = len(self.z)
        assert num_z == 1 and self.z[0] == z_slice

        # build maps from coordinates to indices
        x_map = dict(zip(self.x, np.arange(num_x)))
        y_map = dict(zip(self.y, np.arange(num_y)))
        z_map = dict(zip(self.z, np.arange(num_z)))
        ix = np.array([x_map[v] for v in self.x_coord])
        iy = np.array([y_map[v] for v in self.y_coord])
        iz = np.array([z_map[v] for v in self.z_coord])

        # allocate dense tensor
        self.tensor_data = np.zeros((num_x,num_y,num_var,num_time))
        self.var_mode = 2 # mode for variables

        # read each nodal variable and store in tensor
        # this would be faster if we could assume an ordering for x,y,z
        for v in range(num_var):
            var_name = 'vals_nod_var' + str(v+1)
            nodal_var = np.array(d.variables[var_name])
            for n in range(num_node):
                self.tensor_data[ix[n],iy[n],v,:] = nodal_var[:,valid_ind[n]]

        # read variable names
        tmp = np.array(d.variables['name_nod_var'])
        self.var_name = [str(nc.chartostring(tmp[v,:])) for v in range(num_var)]

        # node and connectivity info
        self.node_ind = np.column_stack((ix,iy,iz))
        self.tensor_node_ind = np.column_stack((ix,iy))
        self.elem_ind = np.array(d.variables['connect1'])[:,0:4]-1 # Assuming Quad's

        # setup info for integration across the mesh
        self.setupIntegrationInfo()

    def setupIntegrationInfo(self):
        """
        Build finite-element quadrature tables and index maps for a planar 2D quad mesh.

        Preconditions
        -------------
        A prior call to `read_sheet(...)` must have succeeded, so that the following
        attributes are defined on `self`:
        
        - self.x, self.y, self.z: 1D arrays of coordinates with len(self.z)==1  
        - self.x_coord, self.y_coord, self.z_coord: full nodal coords on the slice  
        - self.node_ind: array of shape (num_nodes, 3) giving (ix,iy,iz) per node  
        - self.tensor_node_ind: array of shape (num_nodes, 2) giving (ix,iy) per node  
        - self.elem_ind: array of shape (num_elem, 4) with zero-based quad connectivity  
        
        In other words, you must have a planar slice (constant z) and a quad mesh
        (4-node bilinear elements) before calling this method.

        Postconditions
        --------------
        The following attributes will be created or overwritten on `self`:

        A : ndarray, shape (num_qp_per_elem, num_nodes_per_elem)
            Values of the 4 bilinear shape functions at the 2-by-2 Gauss points.
        gp : ndarray, shape (num_qp_per_elem, 2)
            Reference quad coordinates (ξ,η) of the 2-by-2 Gauss points: [±1/√3].
        linear_elem_ind : ndarray, shape (num_qp_per_elem * num_elem,)
            Flattened global node-indices (into `node_linear_ind`) for each element & gp.
        node_linear_ind : ndarray, shape (nx*ny,)
            Maps each grid-node (ix,iy) into a single flat index (F-ordering).
        tensor_elem_linear_ind : ndarray, shape (num_elem, num_nodes_per_elem)
            For each quad element, the flat index of its 4 corner nodes.
        w_det_J : ndarray, shape (num_qp_per_elem, num_elem)
            Quadrature weight -by- |det(J)| at each (gp,element) for use in integration.
        ind : ndarray, shape (num_qp_per_elem * num_elem,)
            Inverse-index array for reassembling element-wise contributions.
        """

        num_x = len(self.x)
        num_y = len(self.y)

        # basis functions evaluated at quadrature points
        oor3 = 1/math.sqrt(3)
        op = 0.5*(1+oor3)
        om = 0.5*(1-oor3)
        self.A = np.array([[op, om], [om, op]])
        self.A = np.kron(self.A,self.A)
        self.gp = np.array([ [-oor3, -oor3],
                             [ oor3, -oor3],
                             [-oor3,  oor3],
                             [ oor3,  oor3] ])
        
        # reorder node indices in each element to match ours (exodus uses a
        # counter-clockwise ordering but we are using tensor product ordering.
        # use 'C' ordering here so nodes in same element are consecutive and avoid a transpose
        self.linear_elem_ind = self.elem_ind[:,[0, 1, 3, 2]]
        self.linear_elem_ind = np.reshape(self.linear_elem_ind,(-1,),order='C')
        self.node_linear_ind = np.ravel_multi_index((self.tensor_node_ind[:,0], self.tensor_node_ind[:,1]), (num_x, num_y), order='F')
        self.tensor_linear_ind = self.node_linear_ind[self.linear_elem_ind]

        num_elem = self.elem_ind.shape[0]
        num_node_per_elem = self.elem_ind.shape[1]
        num_qp_per_elem = num_node_per_elem
        self.tensor_elem_linear_ind = np.zeros((num_elem,num_node_per_elem),dtype=int)
        for e in range(num_elem):
            i = self.elem_ind[e,[0, 1, 3, 2]]
            ti = self.tensor_node_ind[i,:]
            self.tensor_elem_linear_ind[e,:] = np.ravel_multi_index((ti[:,0], ti[:,1]), (num_x, num_y), order='F')

        # quadrature weights (all 1 for linear Gaussian quadrature)
        w = np.ones(num_qp_per_elem)

        # compute element transformation weights
        xx = np.reshape(self.x_coord[self.linear_elem_ind], (num_qp_per_elem, num_elem), order='F')
        yy = np.reshape(self.y_coord[self.linear_elem_ind], (num_qp_per_elem, num_elem), order='F')
        zz = np.reshape(self.z_coord[self.linear_elem_ind], (num_qp_per_elem, num_elem), order='F')
        self.w_det_J = np.zeros((num_qp_per_elem, num_elem))
        for e in range(num_elem):
            for j in range(num_qp_per_elem):
                self.w_det_J[j,e] = w[j]*self.compute_det_jac(xx[:,e], yy[:,e], zz[:,e], self.gp[j,:])

        # compute indices for summing jacobian contributions
        _, self.ind = np.unique(self.linear_elem_ind, return_inverse=True)
        return

    def compute_det_jac(self, x, y, z, gp):
        """
        Compute the determinant of the Jacobian for one isoparametric quad.

        Parameters
        ----------
        x : ndarray, shape (num_nodes_per_elem,)
            Physical x-coordinates of the element's corner nodes.
        y : ndarray, shape (num_nodes_per_elem,)
            Physical y-coordinates of the corner nodes.
        z : ndarray, shape (num_nodes_per_elem,)
            Physical z-coordinates (unused for planar slice but passed through).
        gp : ndarray, shape (2,)
            The reference (ξ,η) quadrature point in [-1,1]^2.

        Returns
        -------
        detJ : float
            |det(dX/dξ)| for the mapping from reference square to physical element.
        """
        
        omx = 0.5*(1-gp[0])
        opx = 0.5*(1+gp[0])
        ome = 0.5*(1-gp[1])
        ope = 0.5*(1+gp[1])
        A = 0.5 * np.array([[-ome,  ome, -ope, ope],
                            [-omx, -opx,  omx, opx]])
        J = np.linalg.det(A@np.column_stack((x,y)))
        return J
    
    def compute_spatial_integral(self, X, var, time, func=None, deriv=None, compute_func=True, compute_deriv=False) -> tuple[np.floating, np.ndarray]:
        """
        Perform a finite-element-style spatial integral of a user-supplied functional
        (and optionally its derivative) over the planar mesh.

        Parameters
        ----------
        X : ndarray or ttb.ktensor
            The nodal data array.  After slicing, this is expected to be
            shape (nx,ny,num_vars,num_time).  If you pass a ktensor, it is
            converted via `.full().double()`.
        var : sequence of int
            Indices along the variable-mode to include in the integral.
        time : sequence of int
            Indices along the time-mode to include in the integral.
        func : callable
            Called as `f = func(v)` where `v` is an array of shape
            (num_qp, num_elem, num_vars, num_time) giving the pointwise
            values at each quadrature point.  Must return an array `f` of
            shape (num_qp, num_elem, num_time).
        deriv : callable, optional
            If `compute_deriv=True`, called as `g = deriv(v)` and must
            return an array of shape (num_qp, num_elem, num_vars, num_time).
        compute_func : bool, default=True
            If True, evaluate and return the integral of `func`.
        compute_deriv : bool, default=False
            If True, also compute and return the assembled gradient tensor
            `∂p/∂X`, of shape (nx,ny,num_vars,num_time).

        Returns
        -------
        p : ndarray, shape (num_time,)
            The time-series of the spatial integral ∫ func over the domain.
        jac : ndarray
            If `compute_deriv=False`, returns an empty array of shape (0,).
            If `compute_deriv=True`, returns the gradient tensor of shape
            (nx,ny,num_vars,num_time).

        Raises
        ------
        AssertionError
            If the element topology is not a quad (num_nodes_per_elem ∉ {4}).
        ValueError
            If `compute_deriv=True` but `deriv` is None.
        """

        if isinstance(X, ttb.ktensor):
            Xf = X.full().double()
        else:
            Xf = X
        num_elem = self.elem_ind.shape[0]
        num_node_per_elem = self.elem_ind.shape[1]
        num_qp_per_elem = num_node_per_elem
        num_var = len(var)
        num_time = len(time)
        num_dim = math.log2(num_node_per_elem)
        assert num_dim == 2 or num_dim == 3
        nx = Xf.shape[0]
        ny = Xf.shape[1]

        p = np.nan
        jac = np.empty((0,))

        # get nodal values for each element
        I = self.tensor_linear_ind
        Xff = np.reshape(Xf[np.ix_(range(nx),range(ny),var,time)],(nx*ny, num_var, num_time), order='F')
        u = np.reshape(Xff[I,:,:],(num_node_per_elem, num_elem*num_var*num_time), order='F')

        # interpolate to quadrature points
        v = np.reshape(self.A@u,(num_qp_per_elem,num_elem,num_var,num_time), order='F')

        # compute integral value
        if compute_func:
            f = func(v) # evaluate functional
            w_det_J1 = np.tile(np.reshape(self.w_det_J,(num_qp_per_elem,num_elem,1),order='F'),(1,1,num_time)) # compute element transformation weights
            p : np.floating = np.sum(w_det_J1*f,(0,1)) # compute integral

        # compute gradient tensor
        if compute_deriv:
            grad = deriv(v) # evaluate derivative functional
            w_det_J2 = np.tile(np.reshape(self.w_det_J,(num_qp_per_elem,num_elem,1,1),order='F'),(1,1,num_var,num_time)) # compute element transformation weights
            w_det_J_grad = np.reshape(w_det_J2*grad,(num_qp_per_elem,num_elem*num_var*num_time),order='F')
            dpdu = self.A.T@w_det_J_grad # Compute element jacobians
            dpdu = np.reshape(dpdu,(num_node_per_elem,num_elem,num_var,num_time),order='F')

            # scatter element jacobians into gradient tensor

            #
            #   This is the gradient of what function ???
            #
            jac = np.zeros((nx*ny,Xf.shape[2],Xf.shape[3]),order='F')
            for e in range(num_elem):
                ti = self.tensor_elem_linear_ind[e,:]
                jac[np.ix_(ti,var,time)] += np.squeeze(dpdu[:,e,:,:])
            jac = np.reshape(jac,Xf.shape,order='F')

        return p, jac
        
    def write_sheet(self, X, vars, fname, template_fname):
        """
        Write a new Exodus-II file by copying a template and overwriting the
        nodal variable fields with data from `X`.

        Parameters
        ----------
        X : ndarray, shape either (nx,ny,nvar,ntime) or (nx,ny,nz,nvar,ntime)
            The full nodal data array to write out.
        vars : sequence of int
            The list of variable-mode indices in `X` corresponding to the
            Exodus “vals_nod_var#” fields to overwrite.
        fname : str
            Path to the output netCDF file to create.
        template_fname : str
            Path to an existing Exodus-II file to use as a dimension/attribute
            template.  All global attributes, dimensions, and variables
            other than `vals_nod_var*` are copied verbatim.

        Raises
        ------
        ValueError
            If `X.ndim` is not 4 or 5.
        """
        import netCDF4 as nc
        
        with nc.Dataset(template_fname,'r') as src, nc.Dataset(fname, 'w') as dst:
            # copy global attributes all at once via dictionary
            dst.setncatts(src.__dict__)
            # copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
            # copy all file data
            for name, variable in src.variables.items():
                x = dst.createVariable(name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
                # copy values
                dst[name][:] = src[name][:]

            # collapse spatial dimensions into a single node dimension
            num_var = len(vars)
            num_time = len(src.dimensions['time_step'])
            num_node = len(src.dimensions['num_nodes'])
            data = np.zeros((num_var, num_time, num_node))
            for n in range(num_node):
                ix, iy, iz = self.node_ind[n]
                if   X.ndim == 5:
                    # X has shape (nx, ny, nz, nvar, ntime)
                    data[:,:,n] = X[ix, iy, iz, :, :]
                elif X.ndim == 4:
                    # X has shape (nx, ny, nvar, ntime) -- drop the singleton z
                    data[:,:,n] = X[ix, iy, :, :]
                else:
                    msg = f"Unexpected X.ndim={X.ndim} in write_sheet; must be 4 or 5."
                    raise ValueError(msg)
            # write data to variables
            for i in range(num_var):
                v = vars[i]
                var_name = 'vals_nod_var' + str(v+1)
                dst[var_name][:] = data[i,:,:][:]
            dst.sync()
