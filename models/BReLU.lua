local BReLU , parent = torch.class('nn.BReLU','nn.Module')

function BReLU:__init()
  parent.__init(self)
  self.elu = nn.ReLU()
  self.buffer = torch.Tensor()
  self.mask = torch.Tensor()
end

function BReLU:updateOutput(input)
  
  if self.mask:nElement() ~= input:nElement()/input:size(1) then
    --print('Init BELU mask')
    local s = input:size()
    s[1] = 1
    self.mask:resize(s)
    self.mask:bernoulli(0.5):mul(2):add(-1)
    self.mask:fill(1)
    self.mask[{{},{1,s[2]/2}}]:fill(-1)
  end

  local mask
  if input:dim() == 4 then
    mask = self.mask:repeatTensor(input:size(1),1,1,1)
  else
    mask = self.mask:repeatTensor(input:size(1),1)
  end

  self.buffer:resizeAs(input):copy(input)
  self.buffer:cmul(mask)
  self.elu:forward(self.buffer)
  self.output:resizeAs(self.elu.output):copy(self.elu.output)
  self.output:cmul(mask)
  
  return self.output
end

function BReLU:updateGradInput(input, gradOutput)
  
  local mask
  if input:dim() == 4 then
    mask = self.mask:repeatTensor(input:size(1),1,1,1)
  else
    mask = self.mask:repeatTensor(input:size(1),1)
  end
  
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:cmul(mask)
  self.elu:backward(self.buffer, self.gradInput)
  self.gradInput:copy(self.elu.gradInput)
  self.gradInput:cmul(mask)
  
  return self.gradInput
end

function BReLU:__tostring__()
   return torch.type(self) .. string.format('()')
end
