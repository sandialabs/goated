import numpy as np
import math

class ExoInfo:
        
    def read_sheet(self, file_name, z_slice=0.0):
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
        x_map = dict(zip(self.x, np.arange(0,num_x)))
        y_map = dict(zip(self.y, np.arange(0,num_y)))
        z_map = dict(zip(self.z, np.arange(0,num_z)))
        ix = np.array([x_map[v] for v in self.x_coord])
        iy = np.array([y_map[v] for v in self.y_coord])
        iz = np.array([z_map[v] for v in self.z_coord])

        # allocate dense tensor
        self.tensor_data = np.zeros((num_x,num_y,num_var,num_time))
        self.var_mode = 2 # mode for variables

        # read each nodal variable and store in tensor
        # this would be faster if we could assume an ordering for x,y,z
        for v in range(0,num_var):
            var_name = 'vals_nod_var' + str(v+1)
            nodal_var = np.array(d.variables[var_name])
            for n in range(0,num_node):
                self.tensor_data[ix[n],iy[n],v,:] = nodal_var[:,valid_ind[n]]

        # read variable names
        tmp = np.array(d.variables['name_nod_var'])
        self.var_name = [str(nc.chartostring(tmp[v,:])) for v in range(0,num_var)]

        # node and connectivity info
        self.node_ind = np.column_stack((ix,iy,iz))
        self.tensor_node_ind = np.column_stack((ix,iy))
        self.elem_ind = np.array(d.variables['connect1'])[:,0:4]-1 # Assuming Quad's

        # setup info for integration across the mesh
        self.setupIntegrationInfo()

    def setupIntegrationInfo(self):

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
        for e in range(0,num_elem):
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
        for e in range(0,num_elem):
            for j in range(0,num_qp_per_elem):
                self.w_det_J[j,e] = w[j]*self.compute_det_jac(xx[:,e], yy[:,e], zz[:,e], self.gp[j,:])

        # compute indices for summing jacobian contributions
        unique_nodes,self.ind = np.unique(self.linear_elem_ind, return_inverse=True)

    def compute_det_jac(self, x, y, z, gp):
        
        omx = 0.5*(1-gp[0])
        opx = 0.5*(1+gp[0])
        ome = 0.5*(1-gp[1])
        ope = 0.5*(1+gp[1])
        A = 0.5 * np.array([[-ome,  ome, -ope, ope],
                            [-omx, -opx,  omx, opx]])
        J = np.linalg.det(A@np.column_stack((x,y)))
        return J
    
    def compute_spatial_integral(self, X, var, time, func=None, deriv=None, compute_func=True, compute_deriv=False):
        import pyttb as ttb

        if isinstance(X,ttb.ktensor):
            Xf = X.full(X).double()
        else:
            Xf = X.double()
        num_elem = self.elem_ind.shape[0]
        num_node_per_elem = self.elem_ind.shape[1]
        num_qp_per_elem = num_node_per_elem
        num_var = len(var)
        num_time = len(time)
        num_dim = math.log2(num_node_per_elem)
        assert num_dim == 2 or num_dim == 3
        nx = Xf.shape[0]
        ny = Xf.shape[1]

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
            p = np.sum(w_det_J1*f,(0,1)) # compute integral

        # compute gradient tensor
        if compute_deriv:
            grad = deriv(v) # evaluate derivative functional
            w_det_J2 = np.tile(np.reshape(self.w_det_J,(num_qp_per_elem,num_elem,1,1),order='F'),(1,1,num_var,num_time)) # compute element transformation weights
            w_det_J_grad = np.reshape(w_det_J2*grad,(num_qp_per_elem,num_elem*num_var*num_time),order='F')
            dpdu = self.A.T@w_det_J_grad # Compute element jacobians
            dpdu = np.reshape(dpdu,(num_node_per_elem,num_elem,num_var,num_time),order='F')

            # scatter element jacobians into gradient tensor
            jac = np.zeros((nx*ny,Xf.shape[2],Xf.shape[3]),order='F')
            for e in range(num_elem):
                ti = self.tensor_elem_linear_ind[e,:]
                jac[np.ix_(ti,var,time)] += np.squeeze(dpdu[:,e,:,:])
            jac = np.reshape(jac,Xf.shape,order='F')

        if compute_func and compute_deriv:
            return p,jac
        elif compute_func:
            return p
        elif compute_deriv:
            return jac
        
    def write_sheet(self, X, vars, fname, template_fname):
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
            for n in range(0,num_node):
                data[:,:,n] = X[self.node_ind[n,0], self.node_ind[n,1], self.node_ind[n,2],:,:]

            # write data to variables
            for i in range(num_var):
                v = vars[i]
                var_name = 'vals_nod_var' + str(v+1)
                dst[var_name][:] = data[i,:,:][:]
            dst.sync()
