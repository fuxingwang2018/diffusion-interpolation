    def _shared_step(self, batch, stage):
         
        x = self.encode(batch)
        edm._sahed_step(batch, stage)