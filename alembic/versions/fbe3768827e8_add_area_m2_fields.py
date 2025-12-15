"""Add area_m2 fields to segmentation_results table

Revision ID: fbe3768827e9
Revises: fbe3768827e8
Create Date: 2025-12-15

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fbe3768827e9'
down_revision = 'fbe3768827e8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Añadir nuevas columnas a la tabla segmentation_results
    op.add_column('segmentation_results', sa.Column('pixel_area_m2', sa.Float(), nullable=False, server_default='1.0'))
    op.add_column('segmentation_results', sa.Column('total_area_m2', sa.Float(), nullable=False, server_default='262144.0'))


def downgrade() -> None:
    # Remover las columnas si se revierte la migración
    op.drop_column('segmentation_results', 'total_area_m2')
    op.drop_column('segmentation_results', 'pixel_area_m2')
