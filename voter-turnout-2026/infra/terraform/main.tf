# [DECISION NEEDED: AWS or Azure?]
# This file is a stub. Fill in the chosen provider block below,
# then delete the other. Run `terraform init` after deciding.
#
# AWS option  → uncomment the aws provider block and resources below
# Azure option → uncomment the azurerm provider block and resources below

# -------------------------------------------------------------------
# VARIABLES (shared across both providers)
# -------------------------------------------------------------------
variable "allowed_office_cidrs" {
  type        = list(string)
  description = "Campaign office IP CIDRs allowed to reach the W&B server."
  default     = ["10.0.0.0/8"] # [DECISION NEEDED: replace with real office IPs]
}

variable "environment" {
  type    = string
  default = "production"
}

# -------------------------------------------------------------------
# AWS OPTION
# Provisions: VPC + private subnet, EC2 for W&B Server, S3 artifact
# backend, security group restricted to campaign office IPs.
# -------------------------------------------------------------------
# Uncomment to use:

# provider "aws" {
#   region = "us-east-1" # [DECISION NEEDED: set your region]
# }
#
# resource "aws_vpc" "main" {
#   cidr_block           = "10.0.0.0/16"
#   enable_dns_hostnames = true
#   tags = { Name = "voter-turnout-vpc" }
# }
#
# resource "aws_subnet" "private" {
#   vpc_id            = aws_vpc.main.id
#   cidr_block        = "10.0.1.0/24"
#   availability_zone = "us-east-1a"
#   tags = { Name = "voter-turnout-private" }
# }
#
# resource "aws_security_group" "wandb" {
#   name   = "wandb-server-sg"
#   vpc_id = aws_vpc.main.id
#
#   ingress {
#     from_port   = 8080
#     to_port     = 8080
#     protocol    = "tcp"
#     cidr_blocks = var.allowed_office_cidrs
#   }
#
#   egress {
#     from_port   = 0
#     to_port     = 0
#     protocol    = "-1"
#     cidr_blocks = ["0.0.0.0/0"]
#   }
# }
#
# resource "aws_instance" "wandb" {
#   ami                    = "ami-0c55b159cbfafe1f0" # [DECISION NEEDED: use latest Amazon Linux 2]
#   instance_type          = "t3.medium"
#   subnet_id              = aws_subnet.private.id
#   vpc_security_group_ids = [aws_security_group.wandb.id]
#
#   user_data = <<-EOF
#     #!/bin/bash
#     docker pull wandb/local
#     docker run -d -p 8080:8080 -v /vol:/vol -e LOCAL_RESTORE=true wandb/local
#   EOF
#
#   tags = { Name = "wandb-server" }
# }
#
# resource "aws_s3_bucket" "artifacts" {
#   bucket = "voter-turnout-wandb-artifacts-${var.environment}"
# }
#
# resource "aws_s3_bucket_versioning" "artifacts" {
#   bucket = aws_s3_bucket.artifacts.id
#   versioning_configuration { status = "Enabled" }
# }
#
# output "wandb_server_url" {
#   value = "http://${aws_instance.wandb.private_ip}:8080"
# }

# -------------------------------------------------------------------
# AZURE OPTION
# Provisions: VNet + private subnet, Linux VM for W&B Server,
# Blob Storage artifact backend, NSG restricted to campaign office IPs.
# -------------------------------------------------------------------
# Uncomment to use:

# provider "azurerm" {
#   features {}
# }
#
# resource "azurerm_resource_group" "main" {
#   name     = "voter-turnout-rg"
#   location = "East US" # [DECISION NEEDED: set your region]
# }
#
# resource "azurerm_virtual_network" "main" {
#   name                = "voter-turnout-vnet"
#   address_space       = ["10.0.0.0/16"]
#   location            = azurerm_resource_group.main.location
#   resource_group_name = azurerm_resource_group.main.name
# }
#
# resource "azurerm_subnet" "private" {
#   name                 = "voter-turnout-subnet"
#   resource_group_name  = azurerm_resource_group.main.name
#   virtual_network_name = azurerm_virtual_network.main.name
#   address_prefixes     = ["10.0.1.0/24"]
# }
#
# resource "azurerm_network_security_group" "wandb" {
#   name                = "wandb-nsg"
#   location            = azurerm_resource_group.main.location
#   resource_group_name = azurerm_resource_group.main.name
#
#   security_rule {
#     name                       = "AllowOffice"
#     priority                   = 100
#     direction                  = "Inbound"
#     access                     = "Allow"
#     protocol                   = "Tcp"
#     source_port_range          = "*"
#     destination_port_range     = "8080"
#     source_address_prefixes    = var.allowed_office_cidrs
#     destination_address_prefix = "*"
#   }
# }
#
# resource "azurerm_linux_virtual_machine" "wandb" {
#   name                = "wandb-server-vm"
#   resource_group_name = azurerm_resource_group.main.name
#   location            = azurerm_resource_group.main.location
#   size                = "Standard_B2s"
#   admin_username      = "azureuser"
#
#   # [DECISION NEEDED: configure SSH public key or admin_password]
#
#   network_interface_ids = [] # wire to NIC referencing the subnet above
#
#   os_disk {
#     caching              = "ReadWrite"
#     storage_account_type = "Standard_LRS"
#   }
#
#   source_image_reference {
#     publisher = "Canonical"
#     offer     = "UbuntuServer"
#     sku       = "22.04-LTS"
#     version   = "latest"
#   }
#
#   custom_data = base64encode(<<-EOF
#     #!/bin/bash
#     docker pull wandb/local
#     docker run -d -p 8080:8080 -v /vol:/vol -e LOCAL_RESTORE=true wandb/local
#   EOF
#   )
# }
#
# resource "azurerm_storage_account" "artifacts" {
#   name                     = "voterturnoutartifacts"
#   resource_group_name      = azurerm_resource_group.main.name
#   location                 = azurerm_resource_group.main.location
#   account_tier             = "Standard"
#   account_replication_type = "LRS"
# }
#
# output "wandb_server_url" {
#   value = "http://<vm-private-ip>:8080" # replace after VM is provisioned
# }
