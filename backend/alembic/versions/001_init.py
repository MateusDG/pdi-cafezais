from alembic import op
import sqlalchemy as sa
from geoalchemy2.types import Geometry

# revision identifiers, used by Alembic.
revision = '001_init'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Enable PostGIS
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table('farm',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('owner', sa.String(length=200), nullable=True),
    )

    op.create_table('plot',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('farm_id', sa.Integer(), sa.ForeignKey('farm.id', ondelete="CASCADE"), nullable=False),
        sa.Column('geom_polygon', Geometry(geometry_type='POLYGON', srid=4326), nullable=True),
        sa.Column('area_ha', sa.Float(), nullable=True),
    )

    op.create_table('image',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('capture_dt', sa.String(length=50), nullable=True),
        sa.Column('gsd_cm', sa.Float(), nullable=True),
        sa.Column('footprint_geom', Geometry(geometry_type='POLYGON', srid=4326), nullable=True),
        sa.Column('uri', sa.String(length=500), nullable=True),
    )

    op.create_table('mosaic',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('method', sa.String(length=50), nullable=True),
        sa.Column('footprint_geom', Geometry(geometry_type='POLYGON', srid=4326), nullable=True),
        sa.Column('uri', sa.String(length=500), nullable=True),
    )

    op.create_table('label',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('plot_id', sa.Integer(), sa.ForeignKey('plot.id', ondelete="CASCADE"), nullable=False),
        sa.Column('geom_polygon', Geometry(geometry_type='POLYGON', srid=4326), nullable=True),
        sa.Column('annotator', sa.String(length=100), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=True),
    )

    op.create_table('model_version',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('params_hash', sa.String(length=64), nullable=True),
        sa.Column('metrics_json', sa.Text(), nullable=True),
    )

    op.create_table('inference_run',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_version_id', sa.Integer(), sa.ForeignKey('model_version.id'), nullable=False),
        sa.Column('mosaic_id', sa.Integer(), sa.ForeignKey('mosaic.id'), nullable=False),
        sa.Column('metrics_json', sa.Text(), nullable=True),
    )

    op.create_table('mask',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('inference_run_id', sa.Integer(), sa.ForeignKey('inference_run.id', ondelete="CASCADE"), nullable=False),
        sa.Column('uri', sa.String(length=500), nullable=True),
        sa.Column('geom_polygon', Geometry(geometry_type='POLYGON', srid=4326), nullable=True),
        sa.Column('area_ha', sa.Float(), nullable=True),
    )

def downgrade():
    op.drop_table('mask')
    op.drop_table('inference_run')
    op.drop_table('model_version')
    op.drop_table('label')
    op.drop_table('mosaic')
    op.drop_table('image')
    op.drop_table('plot')
    op.drop_table('farm')
    op.execute("DROP EXTENSION IF EXISTS postgis")
