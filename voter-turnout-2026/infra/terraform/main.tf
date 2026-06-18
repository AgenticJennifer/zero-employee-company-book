# AWS provider activated.
# To switch to Azure: comment out everything between the AWS markers below
# and uncomment the AZURE OPTION block at the bottom of this file.
#
# Before running:
#   1. Replace allowed_office_cidrs default with real campaign office IPs
#   2. Set your AWS region (default: us-east-1)
#   3. Update the AMI ID to the latest Amazon Linux 2 in your region
#   4. Run: terraform init && terraform plan

# -------------------------------------------------------------------
# VARIABLES
# -------------------------------------------------------------------
variable "allowed_office_cidrs" {
  type        = list(string)
  description = "Campaign office IP CIDRs allowed inbound to the W&B server on port 8080."
  default     = ["10.0.0.0/8"] # TODO: replace with real office IPs before apply
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

# -------------------------------------------------------------------
# AWS OPTION (active)
# Provisions: VPC + private subnet, EC2 t3.medium for W&B Server,
# S3 artifact backend with versioning, SG restricted to office IPs.
# -------------------------------------------------------------------

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = {
    Name        = "voter-turnout-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"
  tags = {
    Name        = "voter-turnout-private"
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "voter-turnout-igw" }
}

resource "aws_security_group" "wandb" {
  name        = "wandb-server-sg"
  description = "Allow W&B traffic from campaign office IPs only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "W&B Server UI from campaign offices"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_office_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "wandb-server-sg", Environment = var.environment }
}

# TODO: replace ami with latest Amazon Linux 2 for your region
# Find with: aws ec2 describe-images --owners amazon --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" --query 'sort_by(Images,&CreationDate)[-1].ImageId'
resource "aws_instance" "wandb" {
  ami                    = "ami-0c55b159cbfafe1f0"
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.private.id
  vpc_security_group_ids = [aws_security_group.wandb.id]

  root_block_device {
    volume_size           = 50
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    amazon-linux-extras install docker -y
    service docker start
    usermod -a -G docker ec2-user
    docker pull wandb/local
    docker run -d \
      -p 8080:8080 \
      -v /vol:/vol \
      -e LOCAL_RESTORE=true \
      --restart unless-stopped \
      wandb/local
  EOF

  tags = { Name = "wandb-server", Environment = var.environment }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "voter-turnout-wandb-artifacts-${var.environment}"
  tags   = { Environment = var.environment }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "wandb_server_private_ip" {
  value       = aws_instance.wandb.private_ip
  description = "Set WANDB_BASE_URL=http://<this-ip>:8080 in your .env"
}

output "wandb_server_url" {
  value = "http://${aws_instance.wandb.private_ip}:8080"
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.id
}

# -------------------------------------------------------------------
# AZURE OPTION (inactive — uncomment to use instead of AWS above)
# Provisions: VNet + subnet, Linux VM, Blob Storage, NSG.
# -------------------------------------------------------------------

# provider "azurerm" {
#   features {}
# }
#
# resource "azurerm_resource_group" "main" {
#   name     = "voter-turnout-rg"
#   location = "East US"
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
#   # TODO: add admin_ssh_key or admin_password block
#   network_interface_ids = [] # TODO: wire to NIC on the subnet above
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
#     docker run -d -p 8080:8080 -v /vol:/vol -e LOCAL_RESTORE=true --restart unless-stopped wandb/local
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
#
#   blob_properties {
#     versioning_enabled = true
#   }
# }
#
# output "wandb_server_url" {
#   value = "http://<vm-private-ip>:8080" # replace with actual IP after apply
# }
