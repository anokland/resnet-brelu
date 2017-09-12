local BELU , parent = torch.class('nn.BELU','nn.Module')

function BELU:__init()
  parent.__init(self)
  self.elu = nn.ELU()
  self.buffer = torch.Tensor()
  self.mask = torch.Tensor()
  self.std = 0
end

function BELU:updateOutput(input)
  
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
  --self.std = self.output:std() + 1e-2
  --self.output:div(self.std)
  
  return self.output
end

function BELU:updateGradInput(input, gradOutput)
  
  local mask
  if input:dim() == 4 then
    mask = self.mask:repeatTensor(input:size(1),1,1,1)
  else
    mask = self.mask:repeatTensor(input:size(1),1)
  end
  
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)--:mul(self.std)
  self.gradInput:cmul(mask)
  self.elu:backward(self.buffer, self.gradInput)
  self.gradInput:copy(self.elu.gradInput)
  self.gradInput:cmul(mask)
  
  return self.gradInput
end

function BELU:__tostring__()
   return torch.type(self) .. string.format('()')
end
